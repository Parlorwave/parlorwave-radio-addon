#!/usr/bin/env python3
"""Is Music Assistant's Autoplay refilling a hosted queue with OLD PARLORWAVE
AUDIO instead of music?

With MA 2.10.6, Autoplay on and mode "auto", MA's refills can be old DJ
breaks and news bulletins -- program audio from days or weeks earlier, or
a break that aired hours ago -- every one a retired ParlorWave token
answering its 4 s silent tombstone. Hygiene removes them as spent, the
shortened queue makes MA refill again with more of the same, and the
queue can end up idle on a tombstone with more ahead and no music at
all.

Why, from MA's own source (2.10.6 player_queues):
  * Autoplay's music refill is seeded from `enqueued_media_items`, a FIFO
    of the last 10 media items passed to `play_media` -- "what the user
    enqueued". Every ParlorWave break or bulletin is inserted with
    `play_media option:"next"`, so after ten of them the listener's own
    seed (a playlist, say) is gone and every seed is speech ("Fetching
    similar tracks ... based on: DJ break, DJ break, ..., <a bulletin's
    title>").
  * Builtin tracks have no similar-tracks provider of their own, so MA asks
    its metadata/plugin providers; Sonic Similarity indexes every track MA
    has analysed -- ParlorWave's old program audio included -- and the
    nearest neighbours of speech are other speech.
  * "auto" falls back to the library mix only when that answer is EMPTY.
`play_media` is the only insert MA offers and it records every track it is
given, so no ParlorWave insert can avoid displacing a seed.

So ParlorWave DETECTS this and SAYS it, and changes nothing in Music
Assistant: it never sets Autoplay or its mode, never adds music, never
restarts or reorders the queue. Hygiene keeps removing the spent items, as
for any spent item ahead of the head. The fix is the listener's: Autoplay
mode "library" or "playlist" in Music Assistant's queue settings, neither
of which seeds from what was enqueued (the library mix holds no ParlorWave
audio -- verified live).

The signature, from one hygiene classification of the future: Autoplay is
on, and at least TAIL_MIN spent ParlorWave program items sit AFTER the last
music item ahead. ParlorWave never inserts a spent item, and a rewind puts
spent items back BETWEEN songs, with music after them; with Autoplay on, MA
keeps the queue topped up, so a spent run at the tail is what its refill
appended. Pure apart from a small in-process registry; no I/O.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import config as _config

logger = logging.getLogger("musicbrain")

#: Spent program items after the last music item that make the finding.
TAIL_MIN = 2

PROBLEM_CODE = "autoplay_refills_program_audio"


def spent_program_tail(classified: list[dict[str, Any]],
                       delete_verdict: str = "delete") -> int:
    """How many positively spent program items come after the last music
    item in this classified future (queue_hygiene.classify_future order).
    With no music ahead at all, every spent program item counts."""
    last_music = -1
    for i, rec in enumerate(classified):
        if not rec.get("kind"):
            last_music = i
    return sum(1 for rec in classified[last_music + 1:]
               if rec.get("kind") and rec.get("verdict") == delete_verdict)


_lock = threading.Lock()
_seen: dict[str, dict[str, Any]] = {}


def observe(queue_id: str, *, autoplay: Any,
            classified: list[dict[str, Any]], at: float,
            delete_verdict: str = "delete") -> int:
    """Record one classification of `queue_id`'s future. Returns the spent
    tail that counted (0 when Autoplay is off or unreported, or the tail is
    below TAIL_MIN). A look that shows music being refilled with no spent
    tail clears an earlier finding; Autoplay switched off clears it too."""
    if not queue_id:
        return 0
    tail = spent_program_tail(classified, delete_verdict)
    music_ahead = any(not rec.get("kind") for rec in classified)
    with _lock:
        if autoplay is not True:
            if autoplay is False:
                _seen.pop(queue_id, None)
            return 0
        if tail >= TAIL_MIN:
            first = queue_id not in _seen
            _seen[queue_id] = {"queue_id": queue_id, "spent_tail": tail,
                               "music_ahead": music_ahead, "at": float(at),
                               "since": (_seen.get(queue_id) or {}).get(
                                   "since", float(at))}
        elif tail == 0 and music_ahead:
            _seen.pop(queue_id, None)
            return 0
        else:
            return 0
    logger.warning(
        "ma_autoplay_program_refill queue=%s spent_after_last_music=%d "
        "music_ahead=%s first=%s (Music Assistant Autoplay is refilling with "
        "old ParlorWave program audio; its seed is the last items enqueued. "
        "Nothing is changed in Music Assistant.)",
        queue_id, tail, music_ahead, first)
    return tail


def current(queue_id: str | None = None) -> list[dict[str, Any]]:
    with _lock:
        rows = [dict(v) for k, v in _seen.items()
                if queue_id is None or k == queue_id]
    return sorted(rows, key=lambda r: r["queue_id"])


def problems() -> list[dict[str, Any]]:
    """One warning per affected queue, in the health report's shape."""
    out = []
    for row in current():
        out.append(_config.problem(
            PROBLEM_CODE, _config.SEVERITY_WARNING, "music_assistant",
            "Music Assistant Autoplay is refilling this queue with old "
            "ParlorWave breaks and news instead of music "
            f"({row['spent_tail']} spent item(s) after the last song). Its "
            "similar-tracks refill is seeded from the last items added to "
            "the queue, and every ParlorWave break and bulletin is one of "
            "them.",
            "In Music Assistant, set this player's Autoplay mode to "
            "'library' or 'playlist' (a music playlist). ParlorWave does not "
            "change Autoplay or add music itself; it keeps removing the "
            "spent items."))
    return out


def reset() -> None:
    with _lock:
        _seen.clear()
