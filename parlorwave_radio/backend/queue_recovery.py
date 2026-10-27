"""One-shot recovery from an MA flow-restart rewind-and-stall,
gated on an OBSERVED causality sequence -- not on a stale idle snapshot.

THE ANOMALY (a repeatable signature): a
high-sample-rate library track (common in many libraries: >44.1 kHz) enters
the 44.1 kHz flow, MA "breaks out to restart flow stream", the restarted
flow resumes at an EARLIER queue position, and the player then lands idle
with the station queue full of unplayed music.

WHY OBSERVED CAUSALITY, NOT A SNAPSHOT. Automatic recovery is risky
because a rewind is indistinguishable from a listener replaying an old
track. Idle-at-a-regressed-position narrows it, but an idle queue that
merely LOOKS historically regressed (e.g. a normal stop early in a long
queue) must never trigger recovery. So recovery requires the observer to
have actually WITNESSED the sequence, live:

    owned station observed PLAYING at a good position
        -> observer sees a substantial ordered-position REGRESSION
           (incident armed in memory: timestamp + last-known-good identity)
        -> queue becomes IDLE within INCIDENT_WINDOW_S of the regression
        -> the last-known-good item still resolves in the live queue
        -> ONE play_index to the first unplayed item after it

No armed incident => no recovery, however idle-and-regressed a snapshot
looks. The incident lives only in memory: a Music Brain restart mid-
anomaly loses the chance to recover, which is preferable to reconstructing
questionable intent from durable state.

INCIDENT WINDOW = 300 s. Rationale: the observer's idle event carries a
120 s debounce, and idle onset follows the rewind only after MA has
stalled for a while; 300 s covers the debounce plus the stall while
staying far too short to attribute an unrelated idle ten minutes later.

CLEARED THE MOMENT the anomaly resolves or the premise changes: healthy
forward playback after the regression (the listener's old-track choice
established itself, or MA recovered on its own), the queue identity
changing, the player pausing / going unavailable / powering off, or the
window expiring. A cleared incident cannot recover.

THE ONE TRANSPORT COMMAND. Every other Music Brain module observes MA's
transport; ma_queue.play_queue_index is the single deliberate exception,
and this module is its only caller (both pinned by test). It can never
skip, pause, seek, reorder, delete, or touch an unowned queue. Kill
switch: playback.auto_recover_idle (default on). Every uncertainty does
nothing and logs; never raises.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger("musicbrain")

INCIDENT_WINDOW_S = 300.0      # regression -> idle must fall inside this
REGRESSION_MIN = 3            # ordered positions; below this is not a rewind
RECOVERY_MIN_GAP_S = 900.0    # belt: >= 15 min between ANY two recoveries

EV_RECOVERED = "ma_idle_rewind_recovered"
EV_ARMED = "ma_rewind_incident_armed"


@dataclass
class Incident:
    queue_id: str
    armed_at: float               # observer clock (epoch seconds)
    good_item_id: str             # last-known-good queue_item_id (pre-rewind)
    good_index: int
    regressed_index: int
    attempted: bool = False


# In-memory only (see module doc). One owned queue at a time.
_lock = threading.Lock()
_incident: Incident | None = None
_last_recovery_at: float = 0.0


def _reset() -> None:
    """Test seam: forget all incident state."""
    global _incident, _last_recovery_at
    with _lock:
        _incident = None
        _last_recovery_at = 0.0


def _owned(queue_id: str, deps: dict[str, Any]) -> bool:
    """Does an ACTIVE ParlorWave station own this queue? The ownership check
    lives here, never in the observer -- ma_observer must not import
    station_store (pinned)."""
    try:
        station = (deps.get("station_fn") or _default_station)()
        return bool(station and station.get("queue_id") == queue_id)
    except Exception:  # noqa: BLE001
        return False


def _default_station():
    import station_store
    return station_store.get_active_station()


def _clear(reason: str) -> None:
    global _incident
    if _incident is not None:
        logger.debug("rewind incident cleared (%s) for %s", reason,
                     _incident.queue_id)
        _incident = None


def observe(prev_snap: dict[str, Any] | None, cur_snap: dict[str, Any],
            events: list[dict[str, Any]], now: float,
            deps: dict[str, Any] | None = None) -> str:
    """Fed one observer poll: maintain the incident and, on an idle event,
    attempt recovery. Returns a short outcome for tests; production logs.
    NEVER raises. `now` is the observer's own clock (epoch seconds), the
    same clock the idle debounce uses, so the window is measured
    consistently."""
    global _incident, _last_recovery_at
    deps = deps or {}
    try:
        with _lock:
            qid = (cur_snap or {}).get("queue_id") or ""
            state = (cur_snap or {}).get("state")

            # Queue identity changed under an armed incident -> clear.
            if _incident is not None and _incident.queue_id != qid:
                _clear("queue_changed")

            # Window expiry.
            if _incident is not None and \
                    now - _incident.armed_at > INCIDENT_WINDOW_S:
                _clear("window_expired")

            # Player left the playing/idle world (paused, gone, off,
            # missing, unavailable, powered down) -> the premise is gone.
            if state not in ("playing", "idle") or not (
                    cur_snap.get("available", True)
                    and cur_snap.get("powered", True)):
                if state != "idle":
                    _clear("not_playing_or_unavailable")

            for ev in events:
                kind = ev.get("kind")
                payload = ev.get("payload") or {}
                if kind == "track_start":
                    steps = payload.get("position_regressed") or 0
                    if steps >= REGRESSION_MIN:
                        # A rewind boundary, live. Arm only if the station
                        # was OWNED and PLAYING just before it.
                        good = (prev_snap or {}).get("item") or {}
                        if (_owned(qid, deps)
                                and (prev_snap or {}).get("state")
                                == "playing"
                                and good.get("queue_item_id")):
                            _incident = Incident(
                                queue_id=qid, armed_at=now,
                                good_item_id=good["queue_item_id"],
                                good_index=int(
                                    (prev_snap or {}).get("current_index")
                                    or 0),
                                regressed_index=int(
                                    cur_snap.get("current_index") or 0))
                            logger.warning(
                                "%s queue=%s regressed %d positions from "
                                "%s (armed; awaiting resolution)",
                                EV_ARMED, qid, steps,
                                _incident.good_item_id[:12])
                    else:
                        # Normal forward boundary AFTER a regression: the
                        # anomaly resolved (MA recovered, or the listener's
                        # chosen track is now playing forward). Rebase.
                        _clear("healthy_progression")
                elif kind == "idle":
                    return self_recover(qid, now, deps)
            return "noted"
    except Exception:  # noqa: BLE001 - fully fenced
        logger.warning("rewind observe failed (nothing done)",
                       exc_info=True)
        return "error"


def self_recover(queue_id: str, now: float, deps: dict[str, Any]) -> str:
    """Idle fired: recover ONLY against an armed, unexpired, owned incident
    whose last-known-good item still resolves. Holds `_lock` (called from
    observe)."""
    global _incident, _last_recovery_at
    inc = _incident
    if inc is None or inc.queue_id != queue_id:
        return "no_incident"
    if inc.attempted:
        return "already_attempted"
    try:
        import settings
        if not bool(settings.load()["playback"].get(
                "auto_recover_idle", True)):
            return "disabled"
    except Exception:  # noqa: BLE001 - unreadable switch defaults ON
        pass
    if now - inc.armed_at > INCIDENT_WINDOW_S:
        _clear("window_expired")
        return "window_expired"
    if now - _last_recovery_at < RECOVERY_MIN_GAP_S and _last_recovery_at:
        return "cooldown"
    if not _owned(queue_id, deps):
        _clear("not_owned")
        return "not_owned"
    import ma_queue
    try:
        items = (deps.get("items_fn") or ma_queue.list_queue_items)(
            queue_id, limit=250, offset=0)
    except Exception:  # noqa: BLE001
        return "queue_unreadable"
    good_pos = next((i for i, it in enumerate(items)
                     if it.get("queue_item_id") == inc.good_item_id), None)
    if good_pos is None:
        # The last-known-good item is gone from the live queue: the queue
        # was rebuilt/changed. Refuse rather than guess.
        _clear("good_item_gone")
        return "good_item_gone"
    resume_at = min(good_pos + 1, len(items) - 1)
    inc.attempted = True
    try:
        (deps.get("play_index_fn") or ma_queue.play_queue_index)(
            queue_id, resume_at)
    except Exception:  # noqa: BLE001
        logger.warning("rewind recovery play_index failed", exc_info=True)
        return "play_failed"
    _last_recovery_at = now
    logger.warning(
        "%s queue=%s: idle after an observed rewind; last-known-good %s "
        "at position %d; resumed at %d -- the one-shot transport exception "
        "(see queue_recovery module doc)", EV_RECOVERED, queue_id,
        inc.good_item_id[:12], good_pos, resume_at)
    return "recovered"
