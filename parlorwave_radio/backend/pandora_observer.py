"""Observer-side Pandora wiring.

Called once per MA-observer cycle, FENCED by the caller (a failure here is
logged and never disturbs observation). It does two bounded things and adds no
autonomous policy:

  1. Classifies the active session's SOURCE MODE from the snapshot's dynamic
     identity, de-duplicated so the work (and the log line) happen on a real
     change, not on every position tick. An inserted local request or ParlorWave
     break never changes the mode, because the mode is the queue's is_dynamic
     identity, not the current item's provenance.

  2. Fires an ALREADY-ARMED local->Pandora handoff at the right boundary: when
     the intended transition item has PLAYED THROUGH and the queue has gone
     idle (its natural end, autoplay off). A pause, an explicit stop, a skip,
     or a source change cancels the pending handoff instead -- a user Stop can
     never trigger it. Firing is one-shot (source_handoff clears the intent
     before the start runs).
"""

from __future__ import annotations

import logging

import source_capabilities as SC
import source_handoff

logger = logging.getLogger("musicbrain")

# the last observed elapsed before a natural end is within this of the item's
# duration (covers one poll interval + a little slack).
_COMPLETION_SLACK_S = 12.0

_last_mode: dict[str, str] = {}


def snapshot_mode(snap: dict) -> str:
    """Source mode from an observer snapshot (is_dynamic first, item provenance
    as the fallback signal, UNKNOWN when neither proves it)."""
    dyn = snap.get("is_dynamic")
    if dyn is True:
        return SC.PANDORA_DYNAMIC_RADIO
    if dyn is False:
        return SC.LOCAL
    item = snap.get("item") or {}
    prov = SC.item_provenance({"uri": item.get("uri"),
                               "provider": item.get("provider")})
    if prov == SC.PANDORA_PROVIDER_MUSIC:
        return SC.PANDORA_DYNAMIC_RADIO
    return SC.UNKNOWN


def _played_through(item: dict, elapsed) -> bool:
    dur = item.get("duration")
    if not (isinstance(dur, (int, float)) and dur > 0):
        return False
    return isinstance(elapsed, (int, float)) and elapsed >= dur - _COMPLETION_SLACK_S


def _default_start(queue_id: str):
    def start(station_uri: str) -> None:
        import ma_queue
        ma_queue.start_station(queue_id, station_uri)
    return start


def observe(prev: dict | None, snap: dict, events: list,
            *, start_fn=None) -> dict:
    """Reflect source mode and drive an armed handoff. Returns a bounded status
    for telemetry (no urls/tokens)."""
    qid = snap.get("queue_id")
    mode = snapshot_mode(snap)
    old_mode = _last_mode.get(qid)
    mode_changed = old_mode != mode
    if mode_changed:
        _last_mode[qid] = mode
        logger.info("pandora source_mode change: queue=%s %s -> %s",
                    qid, old_mode, mode)
        # Only LEAVING an already-active Pandora session invalidates a pending
        # handoff. A handoff lives THROUGH the local phase (mode stays local
        # until it fires), so a local/unknown mode must not cancel it.
        if old_mode == SC.PANDORA_DYNAMIC_RADIO:
            source_handoff.on_source_changed(qid)

    handoff_outcome = None
    h = source_handoff.pending(qid)
    if h is not None and prev is not None:
        prev_item = prev.get("item") or {}
        if (prev_item.get("queue_item_id") == h["trigger_qiid"]
                and prev.get("state") == "playing"):
            state = snap.get("state")
            ended = bool(snap.get("ended"))
            played = _played_through(prev_item, prev.get("elapsed_time"))
            start = start_fn or _default_start(qid)
            if played and (state == "idle" or ended):
                # the armed transition item played THROUGH and the queue idled
                # or MA marked it ended -> fire once (robust to a Cast device
                # that keeps reporting 'playing' at the item's duration)
                handoff_outcome = source_handoff.on_transition_completed(
                    qid, prev_item["queue_item_id"], start_fn=start)
            elif (not played) and state in ("idle", "paused", "gone", "missing"):
                # stopped / paused / skipped BEFORE a clean end -> cancel, never
                # fire (a user Stop must not become a handoff). A pause AT the
                # end leaves the intent armed (neither fires nor cancels).
                source_handoff.cancel(qid, reason=f"transition_{state}")
                handoff_outcome = "cancelled"

    # an explicit stop event also cancels (belt and braces)
    for ev in events:
        if ev.get("kind") == "stop":
            source_handoff.on_stop(qid)

    return {"source_mode": mode, "mode_changed": mode_changed,
            "handoff": handoff_outcome}


def source_status_for(queue_id: str) -> dict:
    """The last classified source mode for a queue (diagnostic)."""
    return {"queue_id": queue_id,
            "source_mode": _last_mode.get(queue_id, SC.UNKNOWN)}


def _reset_for_test() -> None:
    _last_mode.clear()
