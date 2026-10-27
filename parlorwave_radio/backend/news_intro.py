#!/usr/bin/env python3
"""The station line that introduces a publisher news bulletin.

Deliberately NOT written by the LLM. This line has exactly one job -- say
what time it is and who the news is from -- and every failure mode of a
generated line is unacceptable here: an invented headline, a "breaking
news" claim nobody grounded, a weather aside, a callback to a song. A
fixed template set cannot hallucinate, cannot drift, needs no prompt
budget, and renders in TTS time alone.

Two families, chosen by whether the airtime clock is actually known. When
it is not, the line omits the time entirely rather than guessing -- the
same rule the spoken clock follows everywhere else (never render time as
a stand-in for airtime).

Rotation is deterministic on the target hour, so the station does not say
the same sentence every hour, the same hour always produces the same
line (a retry after a failed render is not a different break), and tests
never depend on randomness.
"""

from __future__ import annotations

CLOCK_TEMPLATES: tuple[str, ...] = (
    "It's {clock}. Time for a quick news update from {source}. Coming up now.",
    "{clock} here on {station}. Here's the latest from {source}.",
    "It's {clock}. Let's check in with {source} for the latest news.",
    "{clock}, and here's your news update from {source}.",
)

# Used only when the airtime estimate is unavailable. Never falls back to
# render time: a wrong minute is worse than no minute.
NO_CLOCK_TEMPLATES: tuple[str, ...] = (
    "Time for a quick news update from {source}.",
    "Here's the latest news from {source}.",
    "Let's check in with {source} for the latest news.",
    "Now, a quick news update from {source}.",
)

# What the DJ says when the station never told us who the news is from.
# Deliberately a ROLE, not a publisher: naming any outlet here would put a
# false attribution in the station's own voice the moment somebody points
# the Newsroom at a different feed. "the newsroom" is true of every feed.
DEFAULT_SOURCE = "the newsroom"


def intro_text(clock: str | None, source_label: str | None = None,
               target_hour: str = "", station_name: str | None = None) -> str:
    """The line the DJ speaks before the bulletin.

    clock is the projected AIRTIME already formatted for display ("2:30 PM"),
    or None when it could not be estimated. source_label names the publisher.
    target_hour only selects which template rotates in.
    """
    if station_name is None:
        try:
            import settings
            station_name = settings.resolved_station_name()
        except Exception:
            station_name = "ParlorWave Radio"
    source = (source_label or "").strip() or DEFAULT_SOURCE
    clock = (clock or "").strip()
    pool = CLOCK_TEMPLATES if clock else NO_CLOCK_TEMPLATES
    # Deterministic, stable, and spread across the pool: a plain sum of the
    # hour key's characters is enough and does not vary between processes
    # the way hash() does (PYTHONHASHSEED randomises str hashing).
    idx = sum(ord(c) for c in (target_hour or "")) % len(pool)
    return pool[idx].format(clock=clock, source=source, station=station_name)
