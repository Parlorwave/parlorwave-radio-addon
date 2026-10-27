#!/usr/bin/env python3
"""Has a finite Music Assistant queue genuinely FINISHED -- and when the
listener starts it again, is that a new playback pass?

The failure this prevents: a long queue with Repeat and Autoplay off
plays to its end. Music Assistant's Flow cursor
can fall back to an earlier item with a leftover elapsed time, its
state goes idle, and the ParlorWave screen shows that old song as if it
were paused partway through. Nothing says the queue has ended, so a listener
presses Play -- and MA resumes from the stale cursor, replaying music
the DJ then refuses to host ("a boundary already served").

Two answers, both from what the observer actually saw, never from an idle
state or a stale elapsed time alone:

ENDED needs COMPLETION evidence -- the cursor merely sitting on the last
item is not enough (a listener can stop or pause halfway through the final
song). One of, all from Music Assistant itself:

  ma_ended         MA's own `ended` flag (it sets it only after the last
                   item played to its end: playing/paused -> idle, no next
                   item, the last Flow entry fully streamed or the track
                   within 5 s of its end -- MA 2.10.4 _handle_end_of_queue);
  final_completed  the final item was seen PLAYING within FINAL_TAIL_S of
                   its own duration;
  flow_finished    MA's Flow ran out: with the final item fully streamed
                   (index_in_buffer at the last item) MA's cursor fell back
                   while still "playing" -- its elapsed-over-the-Flow-log
                   position collapses when the stream closes, which is
                   why MA does not set `ended` in that case --
                   and the player went idle within FLOW_FALLBACK_IDLE_S.

...and the player then idle/stopped (not paused, not gone, not unreadable)
on two looks at least CONFIRM_S apart. A stop or pause mid-way through the
final song carries none of the three. It stays ended until something plays
again or music is added after it -- items whose next is ParlorWave's own
program audio are not music (MA's Autoplay can append old breaks).

NEW PASS. Playback starting again after a confirmed ending is a new pass at
once when there is EXPLICIT evidence it was asked for: MA had marked the
queue ended (MA restarts an ended queue only on a play command), or a
person's Play reached this speaker (note_listener_play: the integration
forwards a Home Assistant play service call that carries a user). Without
either, a restart at least USER_RESTART_MIN_S after the ending is presumed
to be the listener's; a sooner one is held as a re-open of the same pass,
and upgraded if the person's Play arrives within LISTENER_INTENT_S. That is
the one unavoidable limitation: a Play from MA's own app, seconds after an
ending MA itself did not mark, cannot be told from MA re-opening a Flow.
A wrap under MA's Repeat mode is also a new pass. Reconnects, retries, a
rewind mid-queue and a pause are the SAME pass, so an automatic rewind can
never turn into an endless station.

Pure: no I/O, no MA calls. ma_observer feeds it one snapshot per look and
publishes the answer; now_playing and dj_engine read it.
"""

from __future__ import annotations

import threading
from typing import Any

#: Two exhausted looks at least this far apart before "ended" is said.
CONFIRM_S = 10.0
#: A restart this long after the ending is the listener's own Play.
USER_RESTART_MIN_S = 60.0
#: A repeated track whose elapsed time went back by more than this started over.
REPEAT_ONE_RESTART_S = 30.0
#: A final item seen playing this close to its own end has completed.
FINAL_TAIL_S = 12.0
#: MA's Flow fallback is an ending only if the player goes idle this soon.
FLOW_FALLBACK_IDLE_S = 60.0
#: How close a person's Play must be to the restart to explain it.
LISTENER_INTENT_S = 30.0

#: Player states that can be a finished queue. `paused` is a pause; `gone`
#: and `missing` are a disconnection (ma_observer.take_snapshot).
_STOPPED = frozenset({"idle", "stopped", "off"})

EV_ENDED = "queue_ended"
EV_NEW_PASS = "new_pass"
EV_REOPENED = "reopened_same_pass"
PASS_RESTART = "restart_after_end"
PASS_REPEAT = "repeat_wrap"

BASIS_MA_ENDED = "ma_ended"
BASIS_FINAL_COMPLETED = "final_completed"
BASIS_FLOW_FINISHED = "flow_finished"

EVIDENCE_MA_ENDED = "ma_ended"
EVIDENCE_LISTENER = "listener_play"
EVIDENCE_PRESUMED = "presumed_after_gap"


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


class QueueEndTracker:
    """One queue's end-of-queue state across observer looks."""

    def __init__(self, queue_id: str):
        self.queue_id = queue_id
        self.last_play: dict[str, Any] | None = None   # the last playing look
        self.completed_at: float | None = None
        self.completed_basis: str | None = None
        self.candidate_since: float | None = None
        self.ended_at: float | None = None
        self.ended_basis: str | None = None
        self.ended_items: int | None = None
        self.reopened_at: float | None = None

    def state(self) -> dict[str, Any]:
        return {"ended": self.ended_at is not None, "since": self.ended_at,
                "basis": self.ended_basis}

    # --- completion evidence, gathered while playing -------------------------
    def _note_completion(self, snap: dict[str, Any], idx: int,
                         at: float) -> None:
        items = _int(snap.get("items"))
        if items is None or not items:
            return
        final = idx >= items - 1
        item = snap.get("item") or {}
        dur = item.get("duration") if isinstance(item, dict) else None
        if dur is None:
            dur = snap.get("duration")
        el = snap.get("elapsed_time")
        if (final and isinstance(dur, (int, float)) and dur > 0
                and isinstance(el, (int, float))
                and float(el) >= float(dur) - FINAL_TAIL_S):
            self.completed_at, self.completed_basis = at, BASIS_FINAL_COMPLETED
            return
        prev = self.last_play
        buf = _int(snap.get("index_in_buffer"))
        if (prev is not None and idx < prev["idx"]
                and buf is not None and buf >= items - 1
                and (prev["idx"] >= items - 1
                     or (prev["buf"] is not None and prev["buf"] >= items - 1))):
            # MA's Flow position collapsed with the last item fully streamed
            self.completed_at, self.completed_basis = at, BASIS_FLOW_FINISHED
            return
        if (self.completed_basis == BASIS_FLOW_FINISHED
                and self.completed_at is not None
                and at - self.completed_at > FLOW_FALLBACK_IDLE_S):
            # it kept playing: that was a listener's jump back, not the end
            self.completed_at = self.completed_basis = None

    def _evidence(self, snap: dict[str, Any]) -> str | None:
        if snap.get("ended") is True:
            return BASIS_MA_ENDED
        if self.completed_basis == BASIS_FLOW_FINISHED and (
                self.completed_at is None
                or float(snap.get("at") or 0.0) - self.completed_at
                > FLOW_FALLBACK_IDLE_S + CONFIRM_S):
            return None
        return self.completed_basis

    # --- one look -----------------------------------------------------------------
    def observe(self, snap: dict[str, Any]) -> dict[str, Any] | None:
        """Feed one look; returns an event dict or None."""
        st = str(snap.get("state") or "")
        at = float(snap.get("at") or 0.0)
        idx = _int(snap.get("current_index"))
        if st == "playing" and idx is not None:
            self.candidate_since = None
            event = None
            if self.ended_at is not None:
                event = self._restart(at)
                self.completed_at = self.completed_basis = None
            elif self._wrapped(snap, idx):
                event = {"kind": EV_NEW_PASS, "reason": PASS_REPEAT,
                         "evidence": "repeat_mode"}
                self.completed_at = self.completed_basis = None
            elif (self.reopened_at is not None
                  and listener_played(self.queue_id, self.reopened_at,
                                      self.reopened_at + LISTENER_INTENT_S)):
                # the person's Play arrived just after the re-open: it was them
                self.reopened_at = None
                event = {"kind": EV_NEW_PASS, "reason": PASS_RESTART,
                         "evidence": EVIDENCE_LISTENER}
            else:
                self._note_completion(snap, idx, at)
            if (self.reopened_at is not None
                    and at - self.reopened_at > LISTENER_INTENT_S):
                self.reopened_at = None
            self.last_play = {"idx": idx, "at": at,
                              "buf": _int(snap.get("index_in_buffer")),
                              "elapsed": snap.get("elapsed_time")}
            return event
        if st not in _STOPPED or idx is None:
            # a pause, a disconnection, an unreadable or absent record: never
            # an ending, and never the end of one already confirmed
            if st == "paused":
                self.candidate_since = None
            return None
        if self.ended_at is not None:
            items = _int(snap.get("items"))
            if (items is not None and self.ended_items is not None
                    and items > self.ended_items and snap.get("ended") is not True
                    and snap.get("next_is_program") is not True):
                # music was added after the end (a person, Autoplay): the
                # queue has something to play again, it is no longer finished.
                # Not when what comes next is ParlorWave's own program audio:
                # MA's Autoplay, seeded from our breaks, can append old breaks
                # and bulletins after the music runs out
                self._clear_end()
            return None
        basis = self._evidence(snap)
        if basis is None:
            self.candidate_since = None       # a stop with music still ahead,
            return None                       # or mid-way through the last song
        if self.candidate_since is None:
            self.candidate_since = at
            return None
        if at - self.candidate_since < CONFIRM_S:
            return None
        self.ended_at, self.ended_basis = at, basis
        self.ended_items = _int(snap.get("items"))
        self.candidate_since = None
        return {"kind": EV_ENDED, "basis": basis}

    def _restart(self, at: float) -> dict[str, Any]:
        """Playback after a confirmed ending: whose was it?"""
        gap = at - (self.ended_at or at)
        basis = self.ended_basis
        self._clear_end()
        if basis == BASIS_MA_ENDED:
            evidence = EVIDENCE_MA_ENDED
        elif listener_played(self.queue_id, at - gap - 1.0, at + 1.0):
            evidence = EVIDENCE_LISTENER
        elif gap >= USER_RESTART_MIN_S:
            evidence = EVIDENCE_PRESUMED
        else:
            self.reopened_at = at
            return {"kind": EV_REOPENED, "ended_for_s": round(gap, 1)}
        return {"kind": EV_NEW_PASS, "reason": PASS_RESTART,
                "evidence": evidence, "ended_for_s": round(gap, 1)}

    def _wrapped(self, snap: dict[str, Any], idx: int) -> bool:
        """MA's Repeat started the queue (or the track) over by itself."""
        mode = str(snap.get("repeat_mode") or "off")
        if mode == "off" or self.last_play is None:
            return False
        last_idx, last_elapsed = self.last_play["idx"], self.last_play["elapsed"]
        if mode == "all":
            items = _int(snap.get("items"))
            return (idx < last_idx and items is not None
                    and last_idx >= items - 1)
        if mode == "one":
            el = snap.get("elapsed_time")
            return (idx == last_idx and isinstance(last_elapsed, (int, float))
                    and isinstance(el, (int, float))
                    and float(last_elapsed) - float(el) > REPEAT_ONE_RESTART_S)
        return False

    def _clear_end(self) -> None:
        self.ended_at = self.ended_basis = self.ended_items = None


# --- the published answer -----------------------------------------------------
_lock = threading.Lock()
_published: dict[str, dict[str, Any]] = {}


def publish(queue_id: str, state: dict[str, Any]) -> None:
    with _lock:
        _published[queue_id] = dict(state)


def current(queue_id: str | None) -> dict[str, Any] | None:
    """The observer's last answer for this queue, or None (no observer)."""
    if not queue_id:
        return None
    with _lock:
        got = _published.get(queue_id)
        return dict(got) if got else None


#: Person-initiated Play requests per queue (epoch seconds), newest last.
_intents: dict[str, list[float]] = {}


def note_listener_play(queue_id: str, at: float) -> None:
    """A person asked this queue's speaker to play (the integration saw a
    Home Assistant play service call with a user). A note, not a command:
    nothing here touches playback."""
    if not queue_id:
        return
    with _lock:
        got = [t for t in _intents.get(queue_id, []) if at - t <= 600.0]
        got.append(float(at))
        _intents[queue_id] = got[-8:]


def listener_played(queue_id: str, start: float, end: float) -> bool:
    with _lock:
        return any(start <= t <= end for t in _intents.get(queue_id, []))


def reset() -> None:
    with _lock:
        _published.clear()
        _intents.clear()
