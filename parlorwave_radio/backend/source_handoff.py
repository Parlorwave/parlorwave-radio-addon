"""Event-driven local -> Pandora handoff executor.

The ONLY supported way to move a local session onto a Pandora dynamic station:

    local final track
      -> ParlorWave transition element (finite)
      -> that transition element COMPLETES
      -> start the already-selected Pandora station (play_media option:"play")
      -> MA creates the dynamic Pandora queue
      -> Pandora A

Starting Pandora is a source-select that REBUILDS the upcoming queue, so the
transition can never be pre-queued in front of Pandora; it must fire exactly
when the INTENDED transition item finishes. This module is the smallest
reusable executor for that: a one-shot, cancellable, failure-safe intent.

Design choices:
  * The trigger is a SPECIFIC queue item completing (its queue_item_id), never
    merely "the player went idle" -- so a user Stop/pause can never fire it.
  * One-shot: the intent is cleared BEFORE the Pandora start runs, so a failure
    cannot re-fire and nothing loops. A failed start leaves transport as it was.
  * A new intent supersedes an old one; a source change / stop / contested
    ownership cancels a pending intent.
  * No durable state file: a handoff is a short-lived in-memory intent. A
    process restart loses at most one pending handoff, which is safe (transport
    is untouched and nothing resumes on its own).

This module performs NO autonomous taste decision: something else decides WHEN
to arm a handoff and WHICH station; the executor only fires the pre-selected
start at the right boundary.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

# outcomes returned by on_transition_completed
FIRED = "fired"
NOT_ARMED = "not_armed"
WRONG_ITEM = "wrong_item"
FAILED = "failed"


@dataclass
class _Handoff:
    trigger_qiid: str
    station_uri: str
    armed_at: str | None


_lock = threading.Lock()
_pending: dict[str, _Handoff] = {}


def arm(queue_id: str, trigger_queue_item_id: str, station_uri: str,
        *, now_iso: str | None = None) -> None:
    """Arm a one-shot handoff: when `trigger_queue_item_id` finishes on
    `queue_id`, start `station_uri`. A new arm supersedes any pending one for
    the same queue (the latest intent wins)."""
    if not (queue_id and trigger_queue_item_id and station_uri):
        raise ValueError("queue_id, trigger item id and station_uri required")
    with _lock:
        _pending[queue_id] = _Handoff(trigger_queue_item_id, station_uri, now_iso)
    logger.info("handoff armed: queue=%s trigger=%s", queue_id,
                trigger_queue_item_id)


def cancel(queue_id: str, reason: str = "cancelled") -> bool:
    """Cancel any pending handoff for a queue. Returns True if one was pending.
    Safe to call for source change, user stop, or contested ownership."""
    with _lock:
        had = _pending.pop(queue_id, None)
    if had is not None:
        logger.info("handoff cancelled: queue=%s reason=%s", queue_id, reason)
    return had is not None


def pending(queue_id: str) -> dict[str, Any] | None:
    """The pending handoff intent for a queue (diagnostic), or None."""
    with _lock:
        h = _pending.get(queue_id)
    return {"trigger_qiid": h.trigger_qiid, "station_uri": h.station_uri,
            "armed_at": h.armed_at} if h else None


def on_transition_completed(queue_id: str, completed_queue_item_id: str, *,
                            start_fn: Callable[[str], Any]) -> str:
    """Feed a completed-item event. If it is the armed trigger item, fire the
    one-shot Pandora start (start_fn(station_uri)) and clear the intent. Any
    other completed item is ignored. Returns one of FIRED / NOT_ARMED /
    WRONG_ITEM / FAILED."""
    with _lock:
        h = _pending.get(queue_id)
        if h is None:
            return NOT_ARMED
        if completed_queue_item_id != h.trigger_qiid:
            return WRONG_ITEM
        # Clear the intent BEFORE starting so a failure cannot re-fire and no
        # second boundary can double-start.
        _pending.pop(queue_id, None)
    try:
        start_fn(h.station_uri)
        logger.info("handoff fired: queue=%s station=%s", queue_id,
                    h.station_uri)
        return FIRED
    except Exception:  # noqa: BLE001 - one meaning: start failed, stay safe
        logger.warning("handoff start failed (transport left as-is): queue=%s",
                       queue_id, exc_info=True)
        return FAILED


def on_stop(queue_id: str) -> None:
    """A user stop/pause cancels a pending handoff -- it must never be mistaken
    for the transition completing."""
    cancel(queue_id, reason="stopped")


def on_source_changed(queue_id: str) -> None:
    """A source/ownership change invalidates a pending handoff."""
    cancel(queue_id, reason="source_changed")


def _reset_for_test() -> None:
    with _lock:
        _pending.clear()
