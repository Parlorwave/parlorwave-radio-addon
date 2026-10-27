#!/usr/bin/env python3
"""Pure measurement of a station's remaining FUTURE queue -- the runway.

No I/O, no clock, no policy. Everything here takes an already-read ordered
future-item list (ma_queue.list_future_items) and answers "what is actually
left", so the numbers are table-testable without a network and so the
question stays separate from what anyone later decides to do about them.

Deliberately NOT here (they belong to station persistence, which does not
exist yet): when to refill, how big a batch is, station identity, generation,
ownership state. `count_owned_future` is the one ownership-adjacent helper,
and it is pure set arithmetic over identities the CALLER supplies -- it
stores nothing and infers nothing.

WHICH FIELDS ARE WHOSE. Music Assistant gives us, per queue item:

    queue_item_id   identity of THIS insertion (32-hex, unique per queue
                    entry -- the same song queued twice has two of them)
    sort_index      DIAGNOSTIC ONLY. It is strictly increasing
                    but NOT contiguous: an option:"add" can produce
                    ...2, 4, 5. Never do arithmetic on it -- no
                    `sort_index > current_index`, no `tail - current`, no
                    `old_tail + 1`. Position comes from the ORDER of the
                    list ma_queue.read_future_items returns, and identity
                    from queue_item_id. (`index` is not a position either;
                    MA reports it as 0 on queue items.)
    media_item.uri  identity of the underlying media (library://track/N)
    duration        seconds, when MA knows them
    available       whether MA believes the item can play

Music Assistant does NOT tell us what is OURS. "This is a DJ break" /
"this is a news bulletin" is a Music Brain verdict, derived by
media_ref.programming_ref matching the rooted path segments of the audio
URLs this process itself serves. There is no MA-native origin or owner
field: `extra_attributes` exists on every item but is empty, and play_media
accepts only URIs, so nothing can be stamped onto a queue item.

UNKNOWN STAYS UNKNOWN. An item we cannot classify is never counted as
music. Runway exists to answer "will the music run out", and the only
dishonest answer is an optimistic one.
"""

from __future__ import annotations

from typing import Any

import media_ref

# Classification verdicts. Three, because there are exactly three answers we
# can honestly give about a queue item.
MUSIC = "music"                # real music, whatever its provider
PROGRAMMING = "programming"    # audio Music Brain itself queued (break/news)
UNKNOWN = "unknown"            # not classifiable from what MA gave us

# MA media_type values that mean "a song". Kept narrow on purpose: a radio
# stream or a podcast episode is real playback but it is not station music,
# and calling it music would inflate a runway that is about to end.
MUSIC_MEDIA_TYPES = frozenset({"track"})


def item_uri(item: dict[str, Any]) -> str:
    """The underlying media URI of a queue item, or ''.

    Read from media_item.uri and nowhere else. For DJ/news audio MA wraps
    our own URL (builtin://track/http://host/broadcast/audio/<token>.mp3) and
    surfaces it here -- which is how break tokens are recognised on air today
    (ma_observer.take_snapshot -> media_ref.break_token_of). No fallback to
    streamdetails is invented: an item with no media_item.uri is honestly
    unclassifiable, and unclassifiable never counts as music."""
    media = item.get("media_item")
    if not isinstance(media, dict):
        return ""
    return str(media.get("uri") or "")


def item_media_type(item: dict[str, Any]) -> str:
    media = item.get("media_item")
    if not isinstance(media, dict):
        return ""
    return str(media.get("media_type") or "").strip().lower()


def is_available(item: dict[str, Any]) -> bool:
    """MA's own availability verdict. Absent is treated as available: the
    field is informational, and refusing to count music because a key was
    missing would understate the runway."""
    return item.get("available") is not False


def item_duration(item: dict[str, Any]) -> float | None:
    """Usable duration in seconds, or None when MA gave us nothing usable.

    None is a real answer and stays one -- a missing duration must never
    become a confident zero, which would silently shorten every minutes-based
    reading. One queued item can be a whole album (a single file of most of an
    hour), so durations are neither uniform nor small."""
    value = item.get("duration")
    if value is None:
        media = item.get("media_item")
        value = media.get("duration") if isinstance(media, dict) else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def classify_item(item: dict[str, Any]) -> tuple[str, str]:
    """(kind, detail) for one queue item. Pure.

    kind is MUSIC / PROGRAMMING / UNKNOWN; detail is the programming kind
    ('break' | 'news') for PROGRAMMING, the reason for UNKNOWN, and '' for
    MUSIC.

    PROGRAMMING IS CHECKED FIRST, deliberately and for the same reason
    playback_lifecycle checks it first: a news bulletin arrives carrying
    every mark of music -- media_type 'track', a real ID3 artist, a
    five-minute duration -- so identity has to win over appearance."""
    uri = item_uri(item)
    if not uri:
        return UNKNOWN, "no_uri"
    kind, _token = media_ref.programming_ref(uri)
    if kind:
        return PROGRAMMING, kind
    media_type = item_media_type(item)
    if not media_type:
        return UNKNOWN, "no_media_type"
    if media_type in MUSIC_MEDIA_TYPES:
        return MUSIC, ""
    return UNKNOWN, f"media_type:{media_type}"


def measure_runway(future_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure an ordered FUTURE item list. Pure; no policy, no thresholds.

    Counts are counts OF THE SUPPLIED LIST. Nothing here is derived from
    sort_index arithmetic -- runway is "how many future items are music",
    answered by classifying each item the caller actually read, never by
    subtracting indexes. Supply the `future` list from
    ma_queue.read_future_items, and check its `complete` flag first: an
    INCOMPLETE read measured here yields a legitimate-looking small number,
    and "we could not see the whole queue" must never be reported as "the
    music is running out".

    Returns counts and time, not a verdict:

        music_count             playable real-music items -- THE runway number
        known_music_seconds     summed duration of those with a usable one
        unknown_duration_count  playable music whose duration MA did not give
        programming_count       DJ break / news audio (never music)
        unknown_count           items we could not classify (never music)
        unavailable_count       items MA marks unavailable (never music)
        total                   items examined

    Count and time are BOTH reported because they diverge materially: one
    queue can hold a whole-album file of most of an hour beside one-minute
    tracks, so "8 tracks left" and "30 minutes left" are different
    questions. Callers decide which one to gate on; this function refuses to
    decide for them.

    unknown_duration_count is what keeps known_music_seconds honest -- it is
    a floor, not a total, and the caller can see exactly how many items are
    missing from it."""
    result = {
        "music_count": 0,
        "known_music_seconds": 0.0,
        "unknown_duration_count": 0,
        "programming_count": 0,
        "unknown_count": 0,
        "unavailable_count": 0,
        "total": 0,
    }
    for item in future_items:
        if not isinstance(item, dict):
            continue
        result["total"] += 1
        kind, _detail = classify_item(item)
        if not is_available(item):
            # Counted once, as unavailable. It is not runway whatever it is:
            # MA does not believe it can play.
            result["unavailable_count"] += 1
            continue
        if kind == PROGRAMMING:
            result["programming_count"] += 1
            continue
        if kind != MUSIC:
            result["unknown_count"] += 1
            continue
        result["music_count"] += 1
        seconds = item_duration(item)
        if seconds is None:
            result["unknown_duration_count"] += 1
        else:
            result["known_music_seconds"] += seconds
    result["known_music_seconds"] = round(result["known_music_seconds"], 3)
    return result


def count_owned_future(future_items: list[dict[str, Any]],
                       owned_queue_item_ids: Any) -> int:
    """How many of the supplied future items carry one of the supplied
    queue_item_ids. Pure set arithmetic -- nothing is persisted, nothing is
    inferred, and this function has no opinion about what ownership means.

    queue_item_id is the right key because it identifies THIS insertion:
    the same song queued twice has two distinct ids, so a user's manual copy
    of a station track can never be miscounted as ours. The ids are unique
    within a queue and stable across repeated reads; they are NOT known to be
    stable across a player/MA reconnect, so a caller that persists them
    needs its own fallback. That decision belongs to station state, not
    here."""
    owned = {str(q) for q in (owned_queue_item_ids or ()) if q}
    if not owned:
        return 0
    return sum(1 for it in future_items
               if isinstance(it, dict)
               and str(it.get("queue_item_id") or "") in owned)


def count_owned_future_by_uri(future_items: list[dict[str, Any]],
                              owned_uris: Any) -> int:
    """Fallback ownership count by media URI, for the one case queue_item_id
    cannot cover: MA regenerating item ids (a reconnect or restart), which
    remains UNPROVEN either way.

    Deliberately a SEPARATE function rather than a fallback branch inside
    count_owned_future, because it is a materially weaker signal and the
    caller must choose it knowingly: a listener who manually queues the same
    song is indistinguishable from our own copy of it here, whereas
    queue_item_id tells them apart exactly. Multiplicity is honoured (a URI
    owned twice counts twice) so this stays a count of ITEMS, not of
    distinct URIs."""
    owned = [str(u) for u in (owned_uris or ()) if u]
    if not owned:
        return 0
    remaining = dict.fromkeys(owned, 0)
    for u in owned:
        remaining[u] += 1
    count = 0
    for it in future_items:
        if not isinstance(it, dict):
            continue
        uri = item_uri(it)
        if remaining.get(uri):
            remaining[uri] -= 1
            count += 1
    return count
