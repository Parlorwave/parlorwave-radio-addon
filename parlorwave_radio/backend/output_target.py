"""Which output does a control action act on when the caller names none?

The failure this prevents: a reseed with no queue_id falling back to
`dj.outputs.enabled[0]`, which may be a silent speaker, so the action
refuses with `nothing_playing` while the owned ParlorWave station is
plainly running on another enabled output. List order would be acting as
target authority.

It is not. `dj.outputs.enabled` says where DJ behaviour is ALLOWED; it does
not say where the station currently is. That question already has an answer --
ma_observer.resolve_target, the same ownership order the observer and the
status headline follow -- and this module is the control-plane wrapper
around it, adding the two things a control action needs and a pure
resolver cannot know: which queues are actually LIVE right now, and which
configured outputs still EXIST in Music Assistant.

Priority when the caller omits a target:

  1. an explicit target (validated, never guessed at)
  2. the active ParlorWave station's queue -- a station is never retargeted
     because a settings list was reordered
  3. a single configured output -- the only place DJ is allowed
  3b. with several allowed, exactly one that is actually playing
  4. more than one plausible target -> target_required, never the first
  5. none -> nothing_playing for an action that needs live playback;
     for a start-from-idle action, the deliberate default output
     (dj.outputs.default_request_output) or a single enabled output

A configured output that no longer exists in MA is never an implicit
target and is reported as unavailable; it is not deleted here -- pruning
configuration is a settings decision, not a routing one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

# ok reasons
EXPLICIT = "explicit"
ACTIVE_STATION = "active_station"
SINGLE_ACTIVE_OUTPUT = "single_active_output"
SELECTED_OUTPUT = "selected_output"
SINGLE_ENABLED_OUTPUT = "single_enabled_output"
# refusals
UNKNOWN_OUTPUT = "unknown_output"
NOTHING_PLAYING = "nothing_playing"
TARGET_REQUIRED = "target_required"
NO_OUTPUT_CONFIGURED = "no_output_configured"

# The states an output must be in to be adopted implicitly. Deliberately
# narrow: existing architecture treats 'playing' as the live station, and
# widening it here would silently change what "active" means everywhere.
LIVE_STATES = ("playing",)


def _load_outputs() -> dict[str, Any]:
    import settings
    return (settings.load().get("dj") or {}).get("outputs") or {}


def _active_sessions(db_path=None) -> list[dict[str, Any]]:
    import dj_engine
    c = dj_engine._connect(db_path)
    try:
        return [{"queue_id": r[0]} for r in c.execute(
            "SELECT queue_id FROM sessions WHERE status = 'active' "
            "AND queue_id IS NOT NULL AND queue_id != ''")]
    finally:
        c.close()


def _inventory(players_fn: Callable[[], Any] | None) -> set[str] | None:
    """Player ids MA currently knows about, or None when MA cannot be read
    (in which case staleness simply cannot be judged and is not guessed)."""
    try:
        import ma_queue
        players = (players_fn or ma_queue.list_players)()
        ids = {p.get("player_id") for p in players if isinstance(p, dict)}
        return {i for i in ids if i} or set()
    except Exception:  # noqa: BLE001
        logger.debug("output inventory unavailable", exc_info=True)
        return None


def _is_live(queue_id: str, state_fn: Callable[[str], Any] | None) -> bool:
    try:
        import ma_queue
        state = (state_fn or ma_queue.queue_state)(queue_id) or {}
        return state.get("state") in LIVE_STATES
    except Exception:  # noqa: BLE001 - an unreadable queue is not a target
        return False


def resolve(explicit: str | None = None, *,
            require_playing: bool = True,
            outputs: dict[str, Any] | None = None,
            sessions_fn: Callable[[], Any] | None = None,
            players_fn: Callable[[], Any] | None = None,
            state_fn: Callable[[str], Any] | None = None,
            db_path=None) -> dict[str, Any]:
    """Resolve the queue a control action should act on.

    Returns {"ok": True, "queue_id", "reason", "unavailable": [...]} or
    {"ok": False, "reason", "candidates": [...], "unavailable": [...]}.
    `require_playing` distinguishes an action that operates on a running
    station from one that may start from idle.
    """
    outs = outputs if outputs is not None else _load_outputs()
    enabled = [o for o in (outs.get("enabled") or []) if o]
    known = _inventory(players_fn)
    unavailable = ([o for o in enabled if o not in known]
                   if known is not None else [])

    if explicit:
        if known is not None and explicit not in known:
            return {"ok": False, "reason": UNKNOWN_OUTPUT,
                    "candidates": [], "unavailable": unavailable}
        return {"ok": True, "queue_id": explicit, "reason": EXPLICIT,
                "unavailable": unavailable}

    # 2. an active station owns the answer, whatever the settings order says
    try:
        import ma_observer
        sessions = (sessions_fn or (lambda: _active_sessions(db_path)))()
        target, reason = ma_observer.resolve_target(sessions, outs, None)
        if reason == ma_observer.REASON_ACTIVE_SESSION and target:
            return {"ok": True, "queue_id": target,
                    "reason": ACTIVE_STATION, "unavailable": unavailable}
        if reason == ma_observer.REASON_AMBIGUOUS:
            live = sorted({(s or {}).get("queue_id") for s in (sessions or [])
                           if (s or {}).get("queue_id")})
            return {"ok": False, "reason": TARGET_REQUIRED,
                    "candidates": live, "unavailable": unavailable}
    except Exception:  # noqa: BLE001
        logger.debug("station binding lookup failed", exc_info=True)

    # a stale id can never be an implicit target
    usable = [o for o in enabled if known is None or o in known]

    # 3. A single configured output is the smallest safe answer and the
    # established contract (ma_observer.resolve_target's own step 3): there
    # is only one place DJ behaviour is allowed, so the operation targets it
    # and its own checks decide whether anything is playing there. Liveness
    # is what DISAMBIGUATES several outputs -- it is not an extra gate on
    # the unambiguous case.
    if len(usable) == 1:
        return {"ok": True, "queue_id": usable[0],
                "reason": SINGLE_ENABLED_OUTPUT, "unavailable": unavailable}

    if require_playing:
        # several allowed outputs: the one actually playing is the station
        live = [o for o in usable if _is_live(o, state_fn)]
        if len(live) == 1:
            return {"ok": True, "queue_id": live[0],
                    "reason": SINGLE_ACTIVE_OUTPUT,
                    "unavailable": unavailable}
        if len(live) > 1:
            return {"ok": False, "reason": TARGET_REQUIRED,
                    "candidates": sorted(live), "unavailable": unavailable}
        if usable:
            return {"ok": False, "reason": NOTHING_PLAYING,
                    "candidates": sorted(usable), "unavailable": unavailable}
        return {"ok": False, "reason": NO_OUTPUT_CONFIGURED,
                "candidates": [], "unavailable": unavailable}

    # 5. start-from-idle: the deliberate default, else a single enabled output
    selected = (outs.get("default_request_output") or "").strip()
    if selected and (known is None or selected in known):
        return {"ok": True, "queue_id": selected, "reason": SELECTED_OUTPUT,
                "unavailable": unavailable}
    if len(usable) > 1:
        return {"ok": False, "reason": TARGET_REQUIRED,
                "candidates": sorted(usable), "unavailable": unavailable}
    return {"ok": False, "reason": NO_OUTPUT_CONFIGURED,
            "candidates": [], "unavailable": unavailable}
