"""The deterministic DJ: typed structural frames for radio copy.

Reads ONLY a BreakPacket -- never settings, a database, the network or a
clock -- and composes a broadcast-quality transition from the facts in it.
Every sentence is drawn from a bank of typed structural frames governed
by slot contracts, daypart affinity, and syntactic repetition suppression.
When the packet holds no track facts it abstains (None).

WHY STRUCTURAL FRAMES.
The deterministic writer is a first-class host with an expanded template
bank and recent-id avoidance. Assembling its breaks through linear sentence
concatenation:
    [Transition Sentence]. [Clock Sentence]. [Weather Sentence].
produces robotic, disconnected data recitals.

So it uses a typed STRUCTURAL FRAME MODEL:
    LEAD + INTERLUDE + HANDOFF

1. COHESIVE SERVICE FRAMES: When both clock and weather are due, it uses a
   unified frame (e.g. `both.jp_ck_wx_nx`) that weaves the ending track,
   the airtime clock, the outdoor condition, and the next track into a
   natural broadcast thought, eliminating clunky period/semicolon glue
   while ensuring 100% of required obligations are voiced.

2. SYNTACTIC DIVERSITY: Frames are typed by `opening_family` (`that_was`,
   `artist_led`, `title_led`, `clock_led`, `station_led`, `weather_led`).
   Rotation actively suppresses reusing the same syntactic lead back-to-back,
   so the station avoids repeating "That was..." on every break.

3. DAYPART AFFINITY: Frames carry optional daypart affinity (`morning`,
   `afternoon`, `evening`, `night`). Morning hours favor crisper,
   clock-friendly openings; night hours favor shorter, unhurried handoffs.

4. SLOT INTEGRITY CONTRACTS: Frames declare station ID and track anchor
   content. A station ID is spoken at most once per break, and partial
   artist-only templates are strictly gated by ANCHOR_NAME_MIN (3 chars).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

from break_packet import BreakPacket, PURPOSE_STATION_REQUEST
import dj_persona

#: The salt on every template seed. Seventeen is arbitrary; what actually
#: keeps the station from repeating itself is the recent-id memory below,
#: and the number only spreads the starting point.
SEED_SALT = 17

#: The shortest name the station's own anchor test can recognise
#: (smart_dj.ANCHOR_NAME_MIN; a test pins them equal). It matters here
#: because bank members may carry a side on its ARTIST alone: a template
#: that drops the title of a song whose artist is unmatchably short would
#: leave a REQUIRED obligation that no checker could ever see paid, which
#: is exactly the failure the floor exists to prevent.
ANCHOR_NAME_MIN = 3

#: How many recently aired template ids a new break steps around.
AVOID_RECENT = 3

#: One break in this many carries an optional (not due) clock, and one in
#: this many an optional station ID. Seeded, so the choice is stable and
#: testable; different salts, so the two do not move together.
CLOCK_EVERY = 3
STATION_ID_EVERY = 5

#: Internal planner vocabulary that must NEVER leak into listener-facing copy.
FORBIDDEN_LEAK_PHRASES: tuple[str, ...] = (
    "the time is due",
    "weather is due",
    "clock is due",
    "service is due",
    "required service",
    "required hosting",
    "hosting requirement",
    "calendar is due",
    "floor break",
    "deterministic fallback",
    "smart dj",
    "i'll hand it over now",
    "hand it over now",
    "smart planner",
    "planner declined",
    "break packet",
    "dj context",
    "hosting obligation",
)


def _check_no_leak(text: str) -> None:
    """Verify that listener-facing copy does not contain internal state phrases."""
    low = text.lower()
    for phrase in FORBIDDEN_LEAK_PHRASES:
        if phrase in low:
            raise ValueError(f"Internal state leak in deterministic copy: {phrase!r}")


def word_target(p: BreakPacket) -> tuple[int, int]:
    """(target_low, target_high) soft word budget for this break,
    consuming authoritative upstream policy via BreakPacket or dj_persona."""
    if p.target_words:
        return p.target_words
    if getattr(p, "purpose", "") == PURPOSE_STATION_REQUEST:
        return (6, 16)
    wb = dj_persona.word_budget(int(p.max_words or 60), getattr(p, "chattiness", None))
    return (wb["target_low"], wb["target_high"])


#: Backwards compatibility alias for callers/tests

soft_word_target = word_target


@dataclass(frozen=True)
class Frame:
    """A typed broadcast structural frame.

    id: Unique identifier, formatted with dotted namespace (e.g. both.jp_nx.wrapped).
    shape: Macro shape category (jp_nx, jp_ck_nx, jp_wx_nx, jp_ck_wx_nx, etc.).
    opening_family: Syntactic lead (that_was, artist_led, title_led, clock_led,
                    station_led, weather_led, you_heard).
    template: Complete sentence or compound sentence template string.
    contains_station: True if template uses {station}.
    contains_time: True if template uses {t}.
    contains_weather: True if template uses {w}.
    clock_sub_id: Clock identifier for composite tracking (e.g. clock.its).
    wx_sub_id: Weather identifier for composite tracking (e.g. wx.outside).
    partial_side: 'next' | 'ending' | None.
    daypart_affinity: Tuple of dayparts this frame naturally suits, or None for universal.
    """
    id: str
    shape: str
    opening_family: str
    template: str
    contains_station: bool = False
    contains_time: bool = False
    contains_weather: bool = False
    clock_sub_id: str | None = None
    wx_sub_id: str | None = None
    partial_side: str | None = None
    daypart_affinity: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# Structural Frame Banks
# ---------------------------------------------------------------------------

#: Core Transition Frames (JP + NX) -- no service payload
JP_NX_FRAMES: tuple[Frame, ...] = (
    # First member is load-bearing: used for the album variant replacement
    Frame("both.that_was.up_next", "jp_nx", "that_was",
          "That was {ea} with {et}. Up next, {na} with {nt}."),
    Frame("both.wrapped.keep_moving", "jp_nx", "artist_led",
          "{ea} just wrapped up {et}. We'll keep it moving with {na} next.",
          daypart_affinity=("morning", "afternoon")),
    Frame("both.from.next_up", "jp_nx", "title_led",
          "That was {et} from {ea}. Next up, {nt} by {na}."),
    Frame("both.you_just_heard", "jp_nx", "you_heard",
          "You just heard {et} from {ea}. Coming up next, {na} with {nt}."),
    Frame("both.closing_out", "jp_nx", "artist_led",
          "{ea} closing things out with {et}. {na} is next, with {nt}."),
    Frame("both.and_here_is", "jp_nx", "that_was",
          "That was {ea} with {et}, and here's {na} with {nt}."),
    Frame("both.stay_right_here", "jp_nx", "artist_led",
          "{ea} there with {et}. We'll stay right here with {nt} from {na}."),
    Frame("both.handing_over", "jp_nx", "artist_led",
          "{ea} wrapping up {et}. Next up, here is {na} with {nt}."),
    # NO padding tier here, on purpose. This pool's frames carry no service
    # payload, so a "richer" member could only be richer in filler --
    # "we've got continuous music lined up", "settle in and keep the music
    # going" -- promises whose only job was reaching a word target. Such
    # tails reach the air, so the class was removed: the word
    # target is SOFT, and a shorter grounded break beats a longer padded
    # one. The one survivor was rewritten down to its grounded remainder.
    Frame("both.jp_nx.moving_forward", "jp_nx", "title_led",
          "{et} from {ea}, right there. Coming up next, {nt} from {na}."),
    Frame("both.ending_led", "jp_nx", "that_was",
          "That was {et}, from {ea}. {na} is up next.",
          partial_side="next"),
    Frame("both.next_led", "jp_nx", "that_was",
          "That was {ea}. Coming up, {na} with {nt}.",
          partial_side="ending"),
)

#: Transition + Clock Frames (JP + CK + NX)
JP_CK_NX_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_ck_nx.mid_clock", "jp_ck_nx", "that_was",
          "That was {ea} with {et}. It's {t}, and {na} is coming up next with {nt}.",
          contains_time=True, clock_sub_id="clock.its"),
    Frame("both.jp_ck_nx.time_is", "jp_ck_nx", "artist_led",
          "{ea} wrapping up {et}. It's {t}. Here's {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.its"),
    Frame("both.jp_ck_nx.front_clock", "jp_ck_nx", "clock_led",
          "It's {t}. You just heard {et} from {ea}, and {na} is next with {nt}.",
          contains_time=True, clock_sub_id="clock.its",
          daypart_affinity=("morning", "afternoon")),
    Frame("both.jp_ck_nx.moving_at", "jp_ck_nx", "title_led",
          "{et} from {ea}. Moving along, it's {t}. Here is {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.moving_at",
          daypart_affinity=("afternoon",)),
    Frame("both.jp_ck_nx.right_at", "jp_ck_nx", "title_led",
          "That was {et} by {ea}. Right now it's {t}, and here is {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.right_at"),
    Frame("both.jp_ck_nx.here_on", "jp_ck_nx", "artist_led",
          "{ea} there with {et}. It's {t} here on {station}. Up next, {na} with {nt}.",
          contains_time=True, contains_station=True, clock_sub_id="clock.here_at"),
    Frame("both.jp_ck_nx.station_sweep", "jp_ck_nx", "clock_led",
          "It's {t} on {station}. That was {et} from {ea}, and {na} is next with {nt}.",
          contains_time=True, contains_station=True, clock_sub_id="clock.here_at",
          daypart_affinity=("morning",)),
    Frame("both.jp_ck_nx.night_roll", "jp_ck_nx", "artist_led",
          "{ea} with {et}. It's {t} on the clock as we roll into {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.moving_at",
          daypart_affinity=("night", "evening")),
    # Richer frames for normal / chatty breaks. Rich means MORE GROUNDED
    # CONTENT (a fuller clock lead, both sides fully named), never a
    # padding clause: "settle in for more music" and "we have plenty more
    # lined up" were removed as filler whose only purpose was
    # length.
    Frame("both.jp_ck_nx.station_flow", "jp_ck_nx", "that_was",
          "That was {ea} with {et}. Checking the clock, it's {t}, as we roll right into {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.moving_at"),
    Frame("both.jp_ck_nx.right_into", "jp_ck_nx", "artist_led",
          "{ea} wrapping up {et}. It's {t}, and we get right into {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.its"),
    Frame("both.jp_ck_nx.time_and_next", "jp_ck_nx", "clock_led",
          "Right now it's {t}. You just heard {et} from {ea}, and up next is {nt} from {na}.",
          contains_time=True, clock_sub_id="clock.right_at"),
    # Compact degradation frames (survives max_words=15)
    Frame("both.jp_ck_nx.compact_next", "jp_ck_nx", "that_was",
          "That was {ea}. It's {t}. {na} is next.",
          contains_time=True, clock_sub_id="clock.its",
          partial_side="next"),
    Frame("both.jp_ck_nx.compact_clock_lead", "jp_ck_nx", "clock_led",
          "It's {t}. That was {ea}. Next up, {na} with {nt}.",
          contains_time=True, clock_sub_id="clock.its",
          partial_side="ending"),
)

#: Transition + Weather Frames (JP + WX + NX)
JP_WX_NX_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_wx_nx.outside", "jp_wx_nx", "that_was",
          "That was {ea} with {et}. Outside right now it's {w}. Up next, {na} with {nt}.",
          contains_weather=True, wx_sub_id="wx.outside"),
    Frame("both.jp_wx_nx.currently", "jp_wx_nx", "artist_led",
          "{ea} just finished {et}. Currently {w} out there. Here's {na} with {nt}.",
          contains_weather=True, wx_sub_id="wx.currently"),
    Frame("both.jp_wx_nx.sitting_at", "jp_wx_nx", "title_led",
          "{et} from {ea}. We're seeing {w} right now. Coming up, {na} with {nt}.",
          contains_weather=True, wx_sub_id="wx.right_now"),
    Frame("both.jp_wx_nx.weather_first", "jp_wx_nx", "weather_led",
          "Currently {w} outside. That was {ea} with {et}. {na} is next with {nt}.",
          contains_weather=True, wx_sub_id="wx.currently"),
    Frame("both.jp_wx_nx.quick_check", "jp_wx_nx", "artist_led",
          "{ea} with {et}. Quick check outside shows {w}. Next up, {na} with {nt}.",
          contains_weather=True, wx_sub_id="wx.outside",
          daypart_affinity=("morning", "afternoon")),
    Frame("both.jp_wx_nx.clean_handoff", "jp_wx_nx", "title_led",
          "That was {et} from {ea}. It's {w} outside. Coming up, {na} with {nt}.",
          contains_weather=True, wx_sub_id="wx.outside"),
    # Richer frames for normal / chatty breaks. The extra content is the
    # STATION ID -- a real service clause -- never a padding promise:
    # "keeping the tunes rolling", "thanks for spending your time" and
    # "stay right where you are" were removed as filler.
    Frame("both.jp_wx_nx.station_flow", "jp_wx_nx", "that_was",
          "That was {ea} with {et}. A quick look outside shows {w}. On {station}, {nt} from {na} is up next.",
          contains_weather=True, contains_station=True, wx_sub_id="wx.outside"),
    Frame("both.jp_wx_nx.conditions_more", "jp_wx_nx", "weather_led",
          "Currently {w} outside. That was {ea} with {et}. This is {station}, and here's {na} with {nt}.",
          contains_weather=True, contains_station=True, wx_sub_id="wx.currently"),
    Frame("both.jp_wx_nx.looking_out", "jp_wx_nx", "artist_led",
          "{ea} wrapping up {et}. Looking outside it's {w}. On {station}, {na} is up next with {nt}.",
          contains_weather=True, contains_station=True, wx_sub_id="wx.outside"),
    # Compact degradation frames (survives max_words=15)
    Frame("both.jp_wx_nx.compact_next", "jp_wx_nx", "that_was",
          "That was {ea}. Currently {w} outside. {na} is next.",
          contains_weather=True, wx_sub_id="wx.currently",
          partial_side="next"),
    Frame("both.jp_wx_nx.compact_wx_lead", "jp_wx_nx", "weather_led",
          "It's {w} outside. That was {ea}. Next up, {na}.",
          contains_weather=True, wx_sub_id="wx.outside",
          partial_side="next"),
)

#: Combined Cohesive Service Frames (JP + CK + WX + NX) -- both services due!
JP_CK_WX_NX_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_ck_wx_nx.mid_combo", "jp_ck_wx_nx", "that_was",
          "That was {ea} with {et}. It's {t}, and {w} outside. Coming up, {na} with {nt}.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.its", wx_sub_id="wx.outside"),
    Frame("both.jp_ck_wx_nx.station_combo", "jp_ck_wx_nx", "artist_led",
          "{ea} just wrapped up {et}. {t} on {station}, and currently {w} outside. Here's {na} with {nt}.",
          contains_time=True, contains_weather=True, contains_station=True,
          clock_sub_id="clock.here_at", wx_sub_id="wx.outside",
          daypart_affinity=("morning", "afternoon")),
    Frame("both.jp_ck_wx_nx.clock_fronted", "jp_ck_wx_nx", "clock_led",
          "It's {t}, and {w} outside. You just heard {et} from {ea}. Coming up, {na} with {nt}.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.its", wx_sub_id="wx.outside",
          daypart_affinity=("morning",)),
    Frame("both.jp_ck_wx_nx.title_lead", "jp_ck_wx_nx", "title_led",
          "{et} from {ea}. It's {t}, and currently {w} outside. We'll keep it moving with {na} next.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.right_at", wx_sub_id="wx.currently"),
    Frame("both.jp_ck_wx_nx.artist_right_now", "jp_ck_wx_nx", "artist_led",
          "{ea} there with {et}. Right now it's {t}, and {w} out there. Up next, {na} with {nt}.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.right_at", wx_sub_id="wx.outside"),
    Frame("both.jp_ck_wx_nx.station_sweep", "jp_ck_wx_nx", "station_led",
          "That was {ea} with {et}. {t} on {station}, where it's {w} outside. Here is {na} with {nt}.",
          contains_time=True, contains_weather=True, contains_station=True,
          clock_sub_id="clock.here_at", wx_sub_id="wx.outside",
          daypart_affinity=("afternoon", "evening")),
    # Richer frames for normal / chatty breaks. Every clause here is a
    # service or a track; the padding tails these frames used to carry
    # ("we've got continuous music lined up for you",
    # "settle in and keep it locked", "thanks for
    # listening") were removed as filler. The handoff itself stays.
    Frame("both.jp_ck_wx_nx.complete_break", "jp_ck_wx_nx", "that_was",
          "That was {ea} with {et}. It's {t} on {station}, and {w} outside. Starting right now, {nt} from {na}.",
          contains_time=True, contains_weather=True, contains_station=True,
          clock_sub_id="clock.here_at", wx_sub_id="wx.outside"),
    Frame("both.jp_ck_wx_nx.service_flow", "jp_ck_wx_nx", "clock_led",
          "It's {t} on {station}, where it's {w} out there. You just heard {et} from {ea}, and here is {na} with {nt}.",
          contains_time=True, contains_weather=True, contains_station=True,
          clock_sub_id="clock.here_at", wx_sub_id="wx.outside"),
    Frame("both.jp_ck_wx_nx.artist_and_service", "jp_ck_wx_nx", "artist_led",
          "{ea} just finished {et}. Checking conditions at {t}, it's {w} outside. Here on {station}, {na} is up next with {nt}.",
          contains_time=True, contains_weather=True, contains_station=True,
          clock_sub_id="clock.right_at", wx_sub_id="wx.outside"),
    # Compact degradation frames (survives max_words=15)
    Frame("both.jp_ck_wx_nx.compact", "jp_ck_wx_nx", "that_was",
          "That was {ea}. It's {t}, and {w} outside. {na} is next.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.its", wx_sub_id="wx.outside",
          partial_side="next"),
    Frame("both.jp_ck_wx_nx.service_first_compact", "jp_ck_wx_nx", "clock_led",
          "It's {t}, and {w} outside. That was {ea}. Next up, {na}.",
          contains_time=True, contains_weather=True,
          clock_sub_id="clock.its", wx_sub_id="wx.outside",
          partial_side="next"),
)

#: Transition + Station ID Frames (JP + ID + NX) -- quiet break brand rotation
JP_ID_NX_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_id_nx.listening_to", "jp_id_nx", "artist_led",
          "{ea} there with {et}. You're listening to {station}. Here's {na} with {nt}.",
          contains_station=True),
    Frame("both.jp_id_nx.this_is", "jp_id_nx", "that_was",
          "That was {et} from {ea}. This is {station}, and {na} is up next with {nt}.",
          contains_station=True),
    Frame("both.jp_id_nx.more_music", "jp_id_nx", "station_led",
          "More music on {station}. That was {ea} with {et}. Coming up, {na} with {nt}.",
          contains_station=True, daypart_affinity=("afternoon", "morning")),
    Frame("both.jp_id_nx.keep_locked", "jp_id_nx", "artist_led",
          "{ea} wrapping up {et}. Keep it locked to {station} for {nt} from {na}.",
          contains_station=True),
)

#: Calendar Transition Frames (JP + CL + NX)
JP_CL_NX_TIMED_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_cl_nx.starts_at", "jp_cl_nx", "that_was",
          "That was {ea} with {et}. On your calendar, {title} starts at {start}. Here's {na} with {nt}."),
    Frame("both.jp_cl_nx.coming_up", "jp_cl_nx", "artist_led",
          "{ea} wrapping up {et}. Coming up today, {title} at {start}. Up next, {na} with {nt}."),
    Frame("both.jp_cl_nx.reminder", "jp_cl_nx", "calendar_led",
          "A quick reminder: {title} starts at {start}. That was {et} from {ea}. {na} is next with {nt}."),
    # Compact degradation frame
    Frame("both.jp_cl_nx.compact_timed", "jp_cl_nx", "calendar_led",
          "On your calendar, {title} at {start}. Here's {na}.",
          partial_side="next"),
)

JP_CL_NX_ALLDAY_FRAMES: tuple[Frame, ...] = (
    Frame("both.jp_cl_nx.allday_on", "jp_cl_nx", "that_was",
          "That was {ea} with {et}. {title} is on your calendar {lead}. Coming up next, {na} with {nt}."),
    Frame("both.jp_cl_nx.allday_have", "jp_cl_nx", "artist_led",
          "{ea} there with {et}. You've got {title} {lead}. Here's {na} with {nt}."),
    # Compact degradation frame
    Frame("both.jp_cl_nx.compact_allday", "jp_cl_nx", "calendar_led",
          "On your calendar, {title} {lead}. Next up, {na}.",
          partial_side="next"),
)


# ---------------------------------------------------------------------------
# Backwards-Compatible Constants & Pools
# ---------------------------------------------------------------------------

#: (template_id, text) for both-sides-known core transitions
BOTH = tuple((f.id, f.template) for f in JP_NX_FRAMES)

#: Bank members that name one side by artist only
PARTIAL = {f.id: f.partial_side for f in JP_NX_FRAMES if f.partial_side}

NEXT_ONLY = (
    ("next.coming_up_next", "Coming up next, {na} with {nt}."),
    ("next.up_next", "Up next, {na} with {nt}."),
    ("next.next_up_by", "Next up, {nt} by {na}."),
    ("next.here_is", "Here's {na} with {nt}."),
    ("next.keep_it_going", "We'll keep it going with {nt} from {na}."),
)

ENDING_ONLY = (
    ("ending.that_was", "That was {ea} with {et}."),
    ("ending.from", "That was {et} from {ea}."),
    ("ending.stay", "{ea} there with {et}. Stay with us."),
    ("ending.just_heard", "You just heard {et} from {ea}."),
    ("ending.wrapped", "{ea} just wrapped up {et}."),
)

#: sparse metadata: {e}/{n} are whatever single fact the track has
SPARSE_BOTH = (
    ("sparse.both.that_was", "That was {e}. Up next, {n}."),
    ("sparse.both.we_heard", "We just heard {e}. Coming up, {n}."),
    ("sparse.both.and_here", "That was {e}, and here's {n}."),
)
SPARSE_ENDING = (
    ("sparse.ending.that_was", "That was {e}."),
    ("sparse.ending.we_heard", "We just heard {e}."),
)
SPARSE_NEXT = (
    ("sparse.next.up_next", "Up next, {n}."),
    ("sparse.next.coming_up", "Coming up, {n}."),
    ("sparse.next.here_is", "Here's {n}."),
)

#: the Start Radio liner
ACK = (
    ("ack.you_got_it", "You got it. More like {artist}, coming right up."),
    ("ack.on_the_way", "More {artist} on the way."),
    ("ack.it_is", "{artist} it is. Keeping it going."),
)
ACK_NO_ARTIST = (("ack.plain", "You got it. Keeping it going."),)

GREETINGS = {
    "morning": "Good morning.",
    "afternoon": "Good afternoon.",
    "evening": "Good evening.",
}

CLOCK = (
    ("clock.its", "It's {t}."),
    ("clock.the_time_is", "The time is {t}."),
    ("clock.here_at", "It's {t} here on {station}."),
)
CLOCK_WITH_ID = "clock.here_at"

STATION_ID = (
    ("id.youre_listening", "You're listening to {station}."),
    ("id.this_is", "This is {station}."),
)

WEATHER = (
    ("wx.right_now", "Right now it's {w}."),
    ("wx.outside", "It's {w} outside."),
    ("wx.currently", "Currently {w}."),
)

CAL_TIMED = (
    ("cal.starts_at", "{title} starts at {start}."),
    ("cal.on_calendar", "On your calendar, {title} at {start}."),
    ("cal.coming_up", "Coming up, {title} at {start}."),
)
CAL_ALLDAY = (
    ("cal.allday_on", "{title} is on your calendar {lead}."),
    ("cal.allday_have", "You've got {title} {lead}."),
)


# ---------------------------------------------------------------------------
# Helper Functions & Repetition Controls
# ---------------------------------------------------------------------------

def _station_named(p: BreakPacket) -> bool:
    """May this break say the station's own name?

    Only when a NEXT TRACK was supplied -- not for style, but because the
    floor never composes copy its own guard will refuse. The unknown-next backstop
    (writer_guard.UNKNOWN_NEXT_NAMED) fires only on a break with no next
    track, and there the station name is a capitalised name that appears
    in no listed fact, so the guard correctly reads it as an invented act.
    """
    return bool(p.next_name)


def _clock_pool(p: BreakPacket) -> tuple[tuple[str, str], ...]:
    return CLOCK if _station_named(p) else tuple(
        m for m in CLOCK if m[0] != CLOCK_WITH_ID)


def _seed(*parts: str) -> int:
    """Stable across processes, well spread, and carrying SEED_SALT."""
    return zlib.crc32("|".join(parts).encode("utf-8")) + SEED_SALT


def _occasional(p: BreakPacket, salt: str, every: int) -> bool:
    """Is this the break that gets that optional element?"""
    if every <= 1:
        return True
    return _seed(p.ending_name, p.next_name, p.station_time, salt) % every == 0


def _opening(text: str, words: int = 3) -> str:
    return " ".join(text.lower().split()[:words])


def _opening_family(text: str) -> str:
    """Determine the syntactic lead family of a text to suppress repetitive leads."""
    low = (text or "").strip().lower()
    if low.startswith(("that was", "that is", "that's")):
        return "that_was"
    if low.startswith(("it's", "the time is", "the time", "checking the clock", "right at")):
        return "clock_led"
    if low.startswith(("you just heard", "we just heard", "you've heard", "we've heard")):
        return "you_heard"
    if low.startswith(("outside right now", "outside", "currently", "quick check outside", "sitting at")):
        return "weather_led"
    if low.startswith(("you're listening", "this is", "more music on", "keep it locked")):
        return "station_led"
    if low.startswith(("on your calendar", "coming up today", "a quick reminder")):
        return "calendar_led"
    if low.startswith(("coming up next", "up next", "next up", "here's", "we'll keep")):
        return "handoff_led"
    return "other"


def _repeats(tpl: str, rendered: str, recent: str) -> bool:
    """Would this member open the way the last aired break opened?"""
    lead = tpl.split("{", 1)[0].strip().lower()
    last = (recent or "").strip().lower()
    if lead:
        return last.startswith(lead)
    return _opening(rendered) == _opening(last)


def _base_id(template_id: Any) -> str:
    """A composite id ('both.jp_nx.wrapped+clock.its') names one transition
    shape plus its service clauses; rotation is about the SHAPE, so only
    the head is compared."""
    return str(template_id or "").split("+", 1)[0]


def _matchable(name: str) -> bool:
    """Is this name long enough for the station's anchor test to see it?"""
    return len("".join(c for c in (name or "").lower() if c.isalnum())) \
        >= ANCHOR_NAME_MIN


def _both_pool(ending_artist: str, next_artist: str) -> tuple[tuple[str, str], ...]:
    """The both-sides-known bank, minus any member that would carry a side
    on an artist name too short to be recognised."""
    pool = tuple(m for m in BOTH
                 if m[0] not in PARTIAL
                 or _matchable(next_artist if PARTIAL[m[0]] == "next"
                               else ending_artist))
    return pool or BOTH


def _label(artist: str, title: str) -> str:
    return artist or title


def _calendar_clause(event: dict[str, Any] | None, seed: int,
                     recent: tuple[str, ...],
                     recent_ids: tuple[str, ...] = ()) -> tuple[str, str] | None:
    """(template_id, sentence) for a DUE calendar event, or None."""
    ev = event or {}
    title = str(ev.get("title") or "").strip()
    if not title:
        return None
    if ev.get("all_day"):
        lead = str(ev.get("lead") or "").strip()
        if not lead:
            return None
        return _pick(CAL_ALLDAY, seed, recent, recent_ids,
                     title=title, lead=lead)
    start = str(ev.get("start_time") or "").strip()
    if not start:
        return None
    return _pick(CAL_TIMED, seed, recent, recent_ids,
                 title=title, start=start)


def _weather_clause(current_weather: str) -> str:
    """The first clause of the listed fact ('78 degrees and sunny')."""
    return (current_weather or "").split(";")[0].strip()


def _pick(pool: tuple[tuple[str, str], ...], seed: int,
          recent: tuple[str, ...], recent_ids: tuple[str, ...] = (),
          **values: str) -> tuple[str, str]:
    """Select from simple string-tuple pools."""
    n = len(pool)
    avoid = {_base_id(i) for i in list(recent_ids or ())[:AVOID_RECENT]}
    last = (recent or ("",))[0] if recent else ""
    start = seed % n
    best: tuple[int, str, str] | None = None
    for k in range(n):
        tid, tpl = pool[(start + k) % n]
        text = tpl.format(**values)
        penalty = (2 if tid in avoid else 0) + (
            1 if last and _repeats(tpl, text, last) else 0)
        if penalty == 0:
            return tid, text
        if best is None or penalty < best[0]:
            best = (penalty, tid, text)
    assert best is not None
    return best[1], best[2]


def _pick_frame(frames: tuple[Frame, ...] | list[Frame],
                seed: int,
                recent: tuple[str, ...],
                recent_ids: tuple[str, ...] = (),
                daypart: str = "",
                can_name_station: bool = True,
                ending_artist: str = "",
                next_artist: str = "",
                target_words: tuple[int, int] | None = None,
                max_words: int | None = None,
                **values: str) -> tuple[Frame, str]:
    """Select the best structural frame from a pool according to rotation,
    opening-family suppression, daypart affinity, budget fit, and safety constraints."""
    candidates = []
    for f in frames:
        if f.contains_station and not can_name_station:
            continue
        if f.partial_side == "next" and not _matchable(next_artist):
            continue
        if f.partial_side == "ending" and not _matchable(ending_artist):
            continue
        candidates.append(f)
    if not candidates:
        candidates = list(frames)

    n = len(candidates)
    avoid = {_base_id(i) for i in list(recent_ids or ())[:AVOID_RECENT]}
    last_text = (recent or ("",))[0] if recent else ""
    last_family = _opening_family(last_text)
    start = seed % n
    best: tuple[int, Frame, str] | None = None

    for k in range(n):
        frame = candidates[(start + k) % n]
        text = frame.template.format(**values)
        penalty = 0
        if _base_id(frame.id) in avoid:
            penalty += 8
        if last_text and _repeats(frame.template, text, last_text):
            penalty += 4
        if last_family and frame.opening_family == last_family:
            penalty += 3
        if daypart and frame.daypart_affinity and daypart not in frame.daypart_affinity:
            penalty += 2

        # Word budget / length affinity
        words = len(text.split())
        if max_words is not None and words > max_words:
            # Over ceiling: penalize heavily, proportional to excess
            penalty += 20 + (words - max_words) * 3
        elif frame.partial_side is not None and (max_words is None or max_words > 18):
            # Compact degradation frames should only be selected under tight budget pressure
            penalty += 10

        # Word budget soft target affinity. Below-target is capped at a
        # NUDGE: an uncapped per-word penalty made the longest frame in a
        # pool beat seed rotation for every break the moment the padding
        # tier was removed -- which is the pressure that bred
        # the padding in the first place. The target is soft affinity; a
        # shorter grounded frame must stay reachable by rotation.
        if target_words is not None:
            t_low, t_high = target_words
            if words < t_low:
                penalty += min(t_low - words, 2)
            elif words > t_high:
                penalty += (words - t_high) * 2

        if penalty == 0:
            return frame, text
        if best is None or penalty < best[0]:
            best = (penalty, frame, text)

    assert best is not None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Request Liners
# ---------------------------------------------------------------------------

def _ack(p: BreakPacket) -> dict[str, Any]:
    artist = str(p.request.get("requested_artist") or "").strip()
    pool = ACK if artist else ACK_NO_ARTIST
    tid, text = _pick(pool, _seed(artist, str(p.request.get("current_song") or "")),
                      p.recent_break_texts, p.recent_template_ids,
                      artist=artist)
    _check_no_leak(text)
    return {"text": text, "template_id": tid,
            "facts_used": ["requested_artist"] if artist else []}


# ---------------------------------------------------------------------------
# Main Deterministic Entrypoint
# ---------------------------------------------------------------------------

def write(p: BreakPacket, station_name: str | None = None) -> dict[str, Any] | None:
    """{'text', 'template_id', 'facts_used'} from the packet alone, or
    None when the packet has nothing truthful to say about the music."""
    if p.purpose == PURPOSE_STATION_REQUEST:
        return _ack(p)
    if station_name is None:
        try:
            import settings
            station_name = settings.resolved_station_name()
        except Exception:
            station_name = "ParlorWave Radio"

    ea, et, na, nt = p.ending_artist, p.ending_title, p.next_artist, p.next_title
    ending_full, next_full = bool(ea and et), bool(na and nt)
    ending_any, next_any = bool(ea or et), bool(na or nt)
    if not ending_any and not next_any:
        return None

    # Service obligations
    clock_due = "station_time" in p.required_service and bool(p.station_time)
    wx = _weather_clause(p.current_weather)
    weather_due = "weather" in p.required_service and bool(wx)
    cal_event = p.calendar_event
    cal_due = cal_event is not None and bool((cal_event or {}).get("title"))
    can_name_station = _station_named(p)
    seed = _seed(p.ending_name, p.next_name)
    service_seed = _seed(p.ending_name, p.next_name, p.station_time)
    recent_ids = p.recent_template_ids
    recent_texts = p.recent_break_texts

    # Authoritative upstream word budget & target
    max_words = int(p.max_words or 60)
    target = word_target(p)

    # -----------------------------------------------------------------------
    # Case A: Both tracks known (the primary radio transition)
    # -----------------------------------------------------------------------
    if ending_full and next_full:
        values = {"ea": ea, "et": et, "na": na, "nt": nt,
                  "t": p.station_time, "w": wx, "station": station_name}

        # Sub-case A1: BOTH Clock and Weather are Due (Cohesive Combined Frame)
        # REQUIRED services NEVER drop: compact frames gracefully absorb budget pressure
        if clock_due and weather_due:
            frame, text = _pick_frame(
                JP_CK_WX_NX_FRAMES, service_seed, recent_texts, recent_ids,
                daypart=p.daypart, can_name_station=can_name_station,
                ending_artist=ea, next_artist=na,
                target_words=target, max_words=max_words, **values)
            tid = f"{frame.id}+{frame.clock_sub_id}+{frame.wx_sub_id}"
            used = ["ending", "next", "station_time", "current_weather"]
            if frame.contains_station:
                used.append("station_id")
            if cal_due:
                c_clause = _calendar_clause(p.calendar_event, service_seed, recent_texts, recent_ids)
                if c_clause:
                    text = f"{text} {c_clause[1]}"
                    used.append("calendar_event")
                    tid = f"{tid}+{c_clause[0]}"
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        # Sub-case A2: Clock Due (Weather Not Due)
        # REQUIRED clock NEVER drops: compact frames degrade music detail under pressure
        if clock_due:
            frame, text = _pick_frame(
                JP_CK_NX_FRAMES, service_seed, recent_texts, recent_ids,
                daypart=p.daypart, can_name_station=can_name_station,
                ending_artist=ea, next_artist=na,
                target_words=target, max_words=max_words, **values)
            tid = f"{frame.id}+{frame.clock_sub_id}"
            used = ["ending", "next", "station_time"]
            if frame.contains_station:
                used.append("station_id")
            if cal_due:
                c_clause = _calendar_clause(p.calendar_event, service_seed, recent_texts, recent_ids)
                if c_clause:
                    text = f"{text} {c_clause[1]}"
                    used.append("calendar_event")
                    tid = f"{tid}+{c_clause[0]}"
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        # Sub-case A3: Weather Due (Clock Not Due)
        # REQUIRED weather NEVER drops: compact frames degrade music detail under pressure
        if weather_due:
            frame, text = _pick_frame(
                JP_WX_NX_FRAMES, service_seed, recent_texts, recent_ids,
                daypart=p.daypart, can_name_station=can_name_station,
                ending_artist=ea, next_artist=na,
                target_words=target, max_words=max_words, **values)
            tid = f"{frame.id}+{frame.wx_sub_id}"
            used = ["ending", "next", "current_weather"]
            if frame.contains_station:
                used.append("station_id")
            if cal_due:
                c_clause = _calendar_clause(p.calendar_event, service_seed, recent_texts, recent_ids)
                if c_clause:
                    text = f"{text} {c_clause[1]}"
                    used.append("calendar_event")
                    tid = f"{tid}+{c_clause[0]}"
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        # Sub-case A4: Calendar Due alone
        if cal_due:
            is_allday = bool((cal_event or {}).get("all_day"))
            pool = JP_CL_NX_ALLDAY_FRAMES if is_allday else JP_CL_NX_TIMED_FRAMES
            title = str((cal_event or {}).get("title") or "")
            start = str((cal_event or {}).get("start_time") or "")
            lead = str((cal_event or {}).get("lead") or "")
            c_values = {**values, "title": title, "start": start, "lead": lead}
            frame, text = _pick_frame(
                pool, seed, recent_texts, recent_ids,
                daypart=p.daypart, can_name_station=can_name_station,
                ending_artist=ea, next_artist=na,
                target_words=target, max_words=max_words, **c_values)
            cal_id = "cal.allday_on" if is_allday else "cal.starts_at"
            tid = f"{frame.id}+{cal_id}"
            used = ["ending", "next", "calendar_event"]
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        # Sub-case A5: Quiet Break (No required service payload)
        colour = None
        if p.chattiness == "high":
            if wx and not weather_due:
                colour = "weather"
            elif p.daypart in GREETINGS and not clock_due:
                colour = "greeting"
            elif p.ending_album and ea and et:
                colour = "album"

        if colour == "album":
            first_frame = JP_NX_FRAMES[0]
            tid = f"{first_frame.id}.album"
            text = first_frame.template.format(**values).replace(
                f"{ea} with {et}.", f"{ea} with {et}, from the album {p.ending_album}.", 1)
            used = ["ending", "next", "ending_album"]
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        if colour == "weather":
            frame, text = _pick_frame(
                JP_WX_NX_FRAMES, seed, recent_texts, recent_ids,
                daypart=p.daypart, can_name_station=can_name_station,
                ending_artist=ea, next_artist=na,
                target_words=target, max_words=max_words, **values)
            tid = f"{frame.id}+{frame.wx_sub_id}"
            used = ["ending", "next", "current_weather"]
            _check_no_leak(text)
            return {"text": text, "template_id": tid, "facts_used": used}

        # Discretionary quiet break with optional station voice & budget shedding
        talkative = p.chattiness != "low"
        opt_clock = talkative and bool(p.station_time) and _occasional(p, "clock", CLOCK_EVERY)
        opt_id = talkative and can_name_station and _occasional(p, "station_id", STATION_ID_EVERY)

        frame, text = _pick_frame(
            JP_NX_FRAMES, seed, recent_texts, recent_ids,
            daypart=p.daypart, can_name_station=can_name_station,
            ending_artist=ea, next_artist=na,
            target_words=target, max_words=max_words, **values)
        tid = frame.id
        parts: list[str | None] = [text]
        optional: list[tuple[int, int]] = []

        clock_id = None
        clock_idx = None
        if opt_clock:
            clock_pool = tuple(m for m in _clock_pool(p) if m[0] != CLOCK_WITH_ID) if frame.contains_station else _clock_pool(p)
            clock_id, c_clause = _pick(clock_pool, service_seed, recent_texts, recent_ids,
                                     t=p.station_time, station=station_name)
            parts.append(c_clause)
            clock_idx = len(parts) - 1
            optional.append((2, clock_idx))
            tid = f"{tid}+{clock_id}"

        id_idx = None
        if opt_id and clock_id != CLOCK_WITH_ID and not frame.contains_station:
            sid, s_clause = _pick(STATION_ID, service_seed, recent_texts, recent_ids,
                                station=station_name)
            parts.append(s_clause)
            id_idx = len(parts) - 1
            optional.append((1, id_idx))
            tid = f"{tid}+{sid}"

        greeting_idx = None
        if colour == "greeting":
            parts.insert(0, GREETINGS[p.daypart])
            greeting_idx = 0
            optional = [(prio, idx + 1) for prio, idx in optional]
            if clock_idx is not None:
                clock_idx += 1
            if id_idx is not None:
                id_idx += 1
            optional.append((0, 0))

        def _render_quiet() -> str:
            return " ".join(x for x in parts if x)

        text = _render_quiet()
        # Word budget priority shedding: optional colour (0), then ID (1), then clock (2)
        effective_budget = max_words
        for _prio, idx in sorted(optional):
            if len(text.split()) <= effective_budget:
                break
            parts[idx] = None
            text = _render_quiet()

        while len(text.split()) > effective_budget and len([x for x in parts if x]) > 1:
            for i in range(len(parts) - 1, -1, -1):
                if parts[i]:
                    parts[i] = None
                    break
            text = _render_quiet()

        final_used = ["ending", "next"]
        if clock_idx is not None and parts[clock_idx] is not None:
            final_used.append("station_time")
        if id_idx is not None and parts[id_idx] is not None:
            final_used.append("station_id")
        if greeting_idx is not None and parts[greeting_idx] is not None:
            final_used.append("daypart")
        if frame.contains_station and "station_id" not in final_used:
            final_used.append("station_id")

        _check_no_leak(text)
        return {"text": text, "template_id": tid, "facts_used": final_used}

    # -----------------------------------------------------------------------
    # Case B: Single-Sided or Sparse Metadata
    # -----------------------------------------------------------------------
    values = {"ea": ea, "et": et, "na": na, "nt": nt}
    if next_full and not ending_any:
        pool, used = NEXT_ONLY, ["next"]
    elif ending_full and not next_any:
        pool, used = ENDING_ONLY, ["ending"]
    else:
        # sparse: at least one track has only one of artist/title
        e, n = _label(ea, et), _label(na, nt)
        values = {"e": e, "n": n}
        if e and n:
            pool, used = SPARSE_BOTH, ["ending", "next"]
        elif e:
            pool, used = SPARSE_ENDING, ["ending"]
        elif n:
            pool, used = SPARSE_NEXT, ["next"]
        else:
            return None

    tid, sentence = _pick(pool, seed, recent_texts, recent_ids, **values)
    parts = [sentence]

    if clock_due:
        clock_id, c_clause = _pick(_clock_pool(p), service_seed, recent_texts, recent_ids,
                                 t=p.station_time, station=station_name)
        parts.append(c_clause)
        used.append("station_time")
        tid = f"{tid}+{clock_id}"

    if weather_due:
        wtid, w_clause = _pick(WEATHER, service_seed, recent_texts, recent_ids, w=wx)
        parts.append(w_clause)
        used.append("current_weather")
        tid = f"{tid}+{wtid}"

    if cal_due:
        c_clause = _calendar_clause(p.calendar_event, service_seed, recent_texts, recent_ids)
        if c_clause:
            parts.append(c_clause[1])
            used.append("calendar_event")
            tid = f"{tid}+{c_clause[0]}"

    text = " ".join(x for x in parts if x)
    _check_no_leak(text)
    return {"text": text, "template_id": tid, "facts_used": used}
