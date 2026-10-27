"""What the Newsroom actually did: a bounded record of bulletin outcomes.

The question this answers is the one a listener could not answer from the
screen: did the half-past bulletin actually air? Before this, the Newsroom
showed the NEXT window and the CURRENT cached item, and a bulletin that
was fetched, armed and then lost its boundary looked exactly like one that
aired -- both left the screen saying the same things a minute later.

WHAT IS RECORDED, and when:

  aired    the bulletin left the air. The only entry that means the
           listener heard it, and the only one Listen Again is offered on.
  skipped  the target passed without airing -- the boundary landed too
           late, the hour was already won by another source, the listener
           paused through the window.
  failed   the feed could not be used inside its window.

WHAT IS NOT RECORDED. A feed TEST writes nothing here: pressing Test is
not the station doing something, and a history full of tests is a history
nobody reads. Nor is an ordinary tick, a watch, or a re-evaluation of an
hour already recorded.

BOUNDED, like every other piece of operational state here. A ring of
MAX_ENTRIES, newest last, in one small JSON document beside the planner's
own. It is not a database, it does not grow, and losing it costs a
listener a list -- never a bulletin.

AUDIO IS NOT RETAINED FOR THIS. Listen Again is offered when the finite
audio the station already downloaded is STILL on disk under its existing
cache policy, and withdrawn when it is not. Nothing here extends a file's
life to keep a button alive.

WHICH IS WHY HISTORY IS LONGER THAN THE AUDIO. The retention rule keeps the
latest bulletin of each source and deletes that source's previous one, so
most rows in this list describe a recording that is gone. A history row is a
record that something happened; it is not a claim that it can be replayed,
and the two are answered separately -- `listen_again` is asked of the disk,
every time the list is read.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import paths
import service_clock

logger = logging.getLogger("musicbrain")

VERSION = 1
MAX_ENTRIES = 60            # a couple of days of two-an-hour; bounded on purpose

AIRED, SKIPPED, FAILED = "aired", "skipped", "failed"
RESULTS = (AIRED, SKIPPED, FAILED)

#: Fields an entry may carry. Closed so a caller cannot quietly file
#: something the screen will never show.
ENTRY_KEYS = ("id", "source_id", "source_name", "item_title", "token",
              "scheduled_for", "aired_at", "duration_s", "result", "reason",
              "recorded_at")


def _path() -> Path:
    return paths.BASE_DIR / "news_history.json"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load() -> list[dict[str, Any]]:
    """Every recorded outcome, oldest first. Fails safe to empty: a missing,
    unreadable, corrupt or future-versioned document reads as 'no history',
    which costs a list and never a bulletin."""
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError,
            UnicodeDecodeError):
        return []
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)][-MAX_ENTRIES:]


def _save(entries: list[dict[str, Any]]) -> None:
    try:
        _atomic_write(_path(), {"version": VERSION,
                                "entries": entries[-MAX_ENTRIES:]})
    except Exception:
        logger.warning("news history write failed (history only)",
                       exc_info=True)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def record(result: str, *, source_id: str | None = None,
           source_name: str | None = None, item_title: str | None = None,
           token: str | None = None, scheduled_for: Any = None,
           aired_at: Any = None, duration_s: Any = None,
           reason: str | None = None, now: datetime | None = None,
           entry_id: str | None = None) -> None:
    """File one outcome. Best-effort and never raises: the Newsroom's
    memory of itself must not be able to interrupt the Newsroom.

    Idempotent per `entry_id` -- the planner can evaluate the same hour
    many times, and an hour that is skipped is often observed skipped on
    several consecutive ticks. Re-filing the same id UPDATES that entry
    rather than appending a second one, so the list stays one row per
    event rather than one row per tick.

    AN EXPLANATION IS NOT OVERWRITTEN BY ITS OWN CONSEQUENCE. Within one
    event, `failed` outranks `skipped`. A window that fails to produce a
    bulletin at all goes on to run out of boundary a few minutes later, and
    last-writer-wins therefore replaced the reason the hour was lost -- an
    actionable sentence about the feed -- with "the boundary landed too
    late", which is true, secondary, and leads whoever reads it away from
    the actual fault. An airing still outranks both: an hour observed
    failing and then airing IS the airing.
    """
    try:
        if result not in RESULTS:
            return
        stamp = now or service_clock.legacy_now()
        entry = {
            "id": str(entry_id or f"{source_id or ''}@{scheduled_for or ''}"
                      or _iso(stamp)),
            "source_id": str(source_id or ""),
            "source_name": str(source_name or ""),
            "item_title": str(item_title or ""),
            "token": str(token or "") or None,
            "scheduled_for": _stamp(scheduled_for),
            "aired_at": _stamp(aired_at),
            "duration_s": _number(duration_s),
            "result": result,
            "reason": str(reason or "") or None,
            "recorded_at": _iso(stamp),
        }
        entries = load()
        for i, existing in enumerate(entries):
            if existing.get("id") == entry["id"]:
                # The same outcome filed again -- the planner re-observing
                # an hour it already recorded -- is not a new fact and is
                # not a new write: the first recording stands, stamp and
                # all. An hour already filed as skipped that later AIRS
                # must become the airing, not stay the skip -- and never
                # both.
                if _same_outcome(existing, entry):
                    return
                if (existing.get("result") == AIRED
                        and entry["result"] != AIRED):
                    # An airing outranks both, as promised above: a late
                    # skip or failure filed under an hour that already aired
                    # (a lost-handoff release, an expiry seen after the
                    # fact) must never rewrite what the listener heard.
                    return
                if (existing.get("result") == FAILED
                        and entry["result"] == SKIPPED):
                    # The diagnosis stands. Nothing is written: the row on
                    # file is already the more specific account of this
                    # event, and rewriting it would only move its timestamp.
                    return
                entries[i] = entry
                break
        else:
            entries.append(entry)
        _save(entries)
    except Exception:
        logger.warning("news history record failed (history only)",
                       exc_info=True)


def _same_outcome(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Equal in everything but when it was recorded."""
    return all(a.get(k) == b.get(k) for k in ENTRY_KEYS if k != "recorded_at")


def _stamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def recent(limit: int = 20, results: tuple[str, ...] | None = None,
           db_path=None) -> list[dict[str, Any]]:
    """The newest outcomes first, each told whether its audio is still
    there to play.

    `listen_again` is decided by ASKING for the file, not by remembering
    that it once existed: the cache is bounded and a bulletin's audio is
    reclaimed on its own schedule, so a button offered from memory would
    be a button that fails when pressed.

    It asks `own_audio_playable`, NOT the transport's `audio_available`.
    The transport answers 200-with-a-tombstone for a retired token on
    purpose, so an old queue row meets a short silence instead of a 410;
    reading that 200 as "there is something to listen to" is how a button
    that plays four seconds of nothing gets onto a screen.
    """
    wanted = tuple(results) if results else RESULTS
    out = []
    for entry in reversed(load()):
        if entry.get("result") not in wanted:
            continue
        row = {k: entry.get(k) for k in ENTRY_KEYS}
        row["listen_again"] = _playable(entry)
        out.append(row)
        if len(out) >= max(1, int(limit or 20)):
            break
    return out


def _playable(entry: dict[str, Any]) -> bool:
    """Is this row's own recording still on disk? Best-effort: history must
    stay readable on an installation whose bulletin module will not import."""
    token = entry.get("token")
    if not token or entry.get("result") != AIRED:
        return False
    try:
        import news_bulletins
        return bool(news_bulletins.own_audio_playable(token))
    except Exception:  # noqa: BLE001 - availability is best-effort
        return False


def latest_per_source() -> list[dict[str, Any]]:
    """The newest recorded outcome for EACH source, newest source first.

    What the Newsroom shows without being asked. The long list behind it is
    still `recent()`; this is the short one, and the retention rule makes it
    the same set of rows that can actually be played -- at most one per
    source, and the one that is the source's own latest.

    A source appears here from the moment it has any outcome at all, airing
    or not: "skipped, the boundary landed too late" is exactly as much of an
    answer to "what has the newsroom been doing" as an airing is."""
    seen: set[str] = set()
    out = []
    for entry in reversed(load()):
        key = str(entry.get("source_id") or entry.get("source_name") or "")
        if key in seen:
            continue
        seen.add(key)
        row = {k: entry.get(k) for k in ENTRY_KEYS}
        row["listen_again"] = _playable(entry)
        out.append(row)
    return out
