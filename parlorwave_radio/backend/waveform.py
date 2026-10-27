"""Normalized waveform abstraction for the currently playing audio track.

Retrieves track RMS energy waveforms from Music Assistant's audio analysis cache
(the 1,800-bin float array computed by the Smart Fades / realtime analysis worker)
and serves it to the Parlorwave Broadcast Console.

Boundaries:
- Pure read-only.
- Never mutates playback, queues, or settings.
- Bounded in-memory LRU cache keyed by (provider, item_id).
- Never exposes Music Assistant tokens or internal network credentials.
- Gracefully handles unanalyzed items, missing streamdetails, and network errors.

A WAVEFORM CAN CHANGE UNDER THE SAME ITEM, so the first answer is not the
answer. MA's `audio_analysis/wave_form` returns the Smart Fades analysis
when one exists and otherwise falls back to any other analysis provider
that stored RMS energy -- and says nothing about which one it returned.
A news bulletin can answer identically early in its playback and
differently later, when (by MA's own description) the Smart Fades
analysis replaces the fallback. Caching the first answer for
provider::item_id would pin the early version for the whole bulletin.

There is no completeness signal to cache on (MA says nothing per item
about which provider answered), so settling is judged from the content
and from what is observed live -- news bulletins revise during
playback, while library tracks and DJ breaks do not. A revision is MA's
one-way
upgrade from a fallback analysis to Smart Fades.

  ACTIVE   a waveform still being asked for is re-read from MA at most
           once every REVALIDATE_S, lazily -- only when a request arrives
           after the interval, never on a timer, so an item nobody is
           looking at costs nothing. A changed answer replaces the cached
           bins and restarts the settle window.
  STABLE   a library item once the SAME bins have been seen for SETTLE_S.
           A programme item (UPGRADE_WATCH_PROVIDERS: the station's own
           finite audio, analysed only once it plays) only SETTLE_S after
           it has been revised; unrevised, it stays active for as long as
           it is asked for. Either way MAX_REVALIDATIONS (10 minutes of
           re-reads, longer than any bulletin) is the hard bound.

Every answer carries `stable` and, while active, `refresh_in_s` -- the
panel's cue for when asking again can return anything new.

EVERY ANSWER IS ALSO AN OBSERVATION. It names the item MA called current,
and carries MA's elapsed_time, the moment MA measured it (elapsed_at, as
the now-playing card stamps it) and the queue state, all from the same
read. The panel uses that to confirm an item before drawing it and to
re-anchor its clock: during a seek MA can report an old queue item as
current for a few hundred milliseconds, and the now-playing sensor can
hold that for a whole refresh cooldown (about ten seconds). A library
track that is already fully analysed settles after SETTLE_S at a cost of
one MA read per REVALIDATE_S until then.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from typing import Any, Callable

import ma_queue

logger = logging.getLogger("musicbrain")

CACHE_CAPACITY = 50
EXPECTED_SAMPLE_COUNT = 1800
REVALIDATE_S = 15.0
SETTLE_S = 60.0
MAX_REVALIDATIONS = 40
UPGRADE_WATCH_PROVIDERS = frozenset({"builtin"})


class WaveformEntry:
    """One item's cached bins and how settled they are.

    `version_since` is when the CURRENT bins were first seen, `checked_at`
    when MA was last asked, `revision` how many times the bins changed."""

    __slots__ = ("bins", "version_since", "checked_at", "revalidations",
                 "revision", "stable")

    def __init__(self, bins: list[float], now: float, stable: bool = False) -> None:
        self.bins = bins
        self.version_since = now
        self.checked_at = now
        self.revalidations = 0
        self.revision = 0
        self.stable = stable


class WaveformCache:
    """Bounded in-memory LRU cache for normalized audio waveforms.

    Keyed by provider and item_id so media items are safely distinguished
    across providers (e.g. filesystem vs builtin URL streams).
    """

    def __init__(self, capacity: int = CACHE_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._cache: collections.OrderedDict[str, WaveformEntry] = collections.OrderedDict()

    def _make_key(self, provider: str, item_id: str) -> str:
        return f"{provider.strip()}::{item_id.strip()}"

    def entry(self, provider: str, item_id: str) -> WaveformEntry | None:
        if not provider or not item_id:
            return None
        key = self._make_key(provider, item_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def get(self, provider: str, item_id: str) -> list[float] | None:
        e = self.entry(provider, item_id)
        return e.bins if e is not None else None

    def put(self, provider: str, item_id: str, entry: WaveformEntry) -> None:
        if not provider or not item_id or not entry.bins:
            return
        key = self._make_key(provider, item_id)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def set(self, provider: str, item_id: str, bins: list[float]) -> None:
        """Store bins already known to be final (a STABLE entry)."""
        self.put(provider, item_id, WaveformEntry(bins, 0.0, stable=True))

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_GLOBAL_CACHE = WaveformCache()


def _sanitize_bins(raw_bins: Any) -> list[float] | None:
    """Validate and sanitize waveform bins from Music Assistant.

    Must be a non-empty sequence of finite floats in [0.0, 1.0].
    Returns sanitized list[float] or None if malformed.
    """
    if not isinstance(raw_bins, (list, tuple)) or not raw_bins:
        return None
    sanitized: list[float] = []
    for val in raw_bins:
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        # Clamp to [0.0, 1.0] defensively
        sanitized.append(max(0.0, min(1.0, f)))
    return sanitized


def get_current_waveform(
    queue_id: str | None = None,
    *,
    deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve normalized waveform metadata for the currently-playing track.

    Returns:
      {"ok": True, "queue_item_id": "...", "source": "music_assistant",
       "sample_count": 1800, "duration_s": 213.4, "bins": [...]}
    Or on failure / unanalyzed:
      {"ok": False, "reason": "not_analyzed"|"no_current_item"|...}
    """
    deps = deps or {}
    raw_queue_fn: Callable[[str], dict[str, Any]] = deps.get("raw_queue_fn") or ma_queue.raw_queue
    api_fn: Callable[..., Any] = deps.get("api_fn") or ma_queue._api
    # NOT `or`: an EMPTY cache defines __len__ == 0 and is falsy, which
    # silently swapped every fresh cache for the shared global one.
    cache: WaveformCache = deps["cache"] if deps.get("cache") is not None else _GLOBAL_CACHE
    resolve_queue_fn: Callable[[], str] = deps.get("resolve_queue_fn")

    resolved_queue_id = (queue_id or "").strip()
    if not resolved_queue_id:
        if resolve_queue_fn:
            resolved_queue_id = resolve_queue_fn()
        else:
            try:
                import app
                resolved_queue_id = app._booth_queue_id()
            except Exception:
                resolved_queue_id = ""

    if not resolved_queue_id:
        return {"ok": False, "reason": "no_queue_target"}

    try:
        queue_obj = raw_queue_fn(resolved_queue_id)
    except Exception as exc:
        logger.debug("Failed to read raw queue for waveform (%s): %s", resolved_queue_id, exc)
        return {"ok": False, "reason": "ma_unavailable", "error": "queue read failed"}

    current_item = (queue_obj or {}).get("current_item")
    if not isinstance(current_item, dict):
        return {"ok": False, "reason": "no_current_item"}

    queue_item_id = str(current_item.get("queue_item_id") or "")
    streamdetails = current_item.get("streamdetails")
    if not isinstance(streamdetails, dict):
        streamdetails = {}

    obs = _observation(queue_obj)
    item_id = str(streamdetails.get("item_id") or "")
    provider = str(streamdetails.get("provider") or "")
    duration_s = float(current_item.get("duration") or streamdetails.get("duration") or 0.0)

    if not item_id or not provider:
        return {
            "ok": False,
            "reason": "no_stream_details",
            "queue_item_id": queue_item_id,
            **obs,
        }

    clock: Callable[[], float] = deps.get("clock") or time.monotonic
    now = clock()
    entry = cache.entry(provider, item_id)

    # 1. Cache: a stable entry always, an active one inside its interval.
    if entry is not None and (entry.stable or now - entry.checked_at < REVALIDATE_S):
        return _ok(queue_item_id, duration_s, entry, now, obs)

    # 2. Ask MA (first sight, or an active entry due for revalidation). A
    # failed re-read keeps serving what we have and waits one interval.
    try:
        raw_waveform = api_fn(
            "audio_analysis/wave_form",
            {"item_id": item_id, "provider_instance_id_or_domain": provider},
        )
    except Exception as exc:
        logger.debug("MA audio_analysis/wave_form query failed: %s", exc)
        if entry is not None:
            entry.checked_at = now
            return _ok(queue_item_id, duration_s, entry, now, obs)
        return {
            "ok": False,
            "reason": "ma_unavailable",
            "queue_item_id": queue_item_id,
            "error": "analysis query failed",
            **obs,
        }

    sanitized = _sanitize_bins(raw_waveform) if raw_waveform is not None else None
    if sanitized is None:
        if entry is not None:
            entry.checked_at = now
            return _ok(queue_item_id, duration_s, entry, now, obs)
        if raw_waveform is None:
            return {
                "ok": False,
                "reason": "not_analyzed",
                "queue_item_id": queue_item_id,
                **obs,
            }
        logger.warning(
            "Malformed waveform payload received from MA for queue item %s",
            queue_item_id[:32],
        )
        return {
            "ok": False,
            "reason": "malformed_waveform",
            "queue_item_id": queue_item_id,
            **obs,
        }

    # 3. Store, or revalidate.
    if entry is None:
        entry = WaveformEntry(sanitized, now)
        cache.put(provider, item_id, entry)
    else:
        entry.revalidations += 1
        entry.checked_at = now
        if sanitized != entry.bins:
            entry.bins = sanitized
            entry.version_since = now
            entry.revision += 1
            logger.info(
                "waveform_revised queue_item=%s revision=%d revalidations=%d",
                queue_item_id[:32], entry.revision, entry.revalidations,
            )
        watch = provider in UPGRADE_WATCH_PROVIDERS and entry.revision == 0
        if ((not watch and now - entry.version_since >= SETTLE_S)
                or entry.revalidations >= MAX_REVALIDATIONS):
            entry.stable = True

    return _ok(queue_item_id, duration_s, entry, now, obs)


def _observation(queue_obj: dict[str, Any]) -> dict[str, Any]:
    """MA's position for the item it calls current, from the same read."""
    import now_playing
    import time as _time
    q = queue_obj or {}
    return {
        "state": q.get("state"),
        "elapsed": q.get("elapsed_time"),
        "elapsed_at": now_playing.elapsed_sampled_at(q, _time.time()),
    }


def _ok(queue_item_id: str, duration_s: float, entry: WaveformEntry,
        now: float, obs: dict[str, Any]) -> dict[str, Any]:
    refresh_in_s = None
    if not entry.stable:
        refresh_in_s = round(max(1.0, REVALIDATE_S - (now - entry.checked_at)), 1)
    return {
        "ok": True,
        "queue_item_id": queue_item_id,
        "source": "music_assistant",
        "sample_count": len(entry.bins),
        "duration_s": duration_s,
        "stable": entry.stable,
        "revision": entry.revision,
        "refresh_in_s": refresh_in_s,
        **obs,
        "bins": entry.bins,
    }
