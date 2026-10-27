#!/usr/bin/env python3
"""Automatic, event-driven queue hygiene: WHEN to ask.

The engine (queue_hygiene.py) decides WHAT is safe to remove from the
future of a Music Assistant queue and does the removing, under every gate
it owns: the playback.queue_hygiene permission, the safety latch, queue
authority, the station mutation lock, the refill / Start Radio exclusions, the
minimum gap, the playing state, the buffer, positive ownership, positive
staleness, the serving belt and the protected-sequence check. This module
adds exactly one thing on top: the lifecycle moments at which the
product asks that engine to look, and a gate so two such requests can
never run the engine at once.

One of those moments also decides WHICH POLICY the pass runs under
(see MANDATORY below): a confirmed regression reconciles Parlorwave's own
spent programming whether or not playback.queue_hygiene permits routine
cleanup. That waives the PERMISSION gate and nothing else.

THE MOMENTS, all observed by the Music Assistant observer -- the first two
where Parlorwave's own aired program items come to sit ahead of the play
head again, the third where spent ones can be left there while nothing
plays, the last two because spent items also turn up where none of the
first three has a reason to look:

    initial_attach       the observer's first stable PLAYING reconcile of
                         an authoritative queue after startup, restart or
                         a target change -- the restart rewind case
    position_regressed   a confirmed backward boundary on the ordered
                         queue (Music Assistant's flow-restart resume, a
                         listener replaying history): the items the
                         station already spoke are in the future again --
                         including a queue that came back EARLIER than the
                         last live position after an outage or a
                         quiet stretch
    quiet_recovery       playing again after a QUIET stretch (the player
                         paused, idle, off or unavailable, or the queue
                         unreadable, for ma_observer.RECOVERY_QUIET_S or
                         longer) at a position that did not go back: breaks
                         and bulletins waiting ahead may have gone terminal
                         while nothing played (a queued break's TTL, a
                         bulletin's age) -- routine, like the attach

    program_boundary     a break or a bulletin (or the tombstone standing
                         in for a retired one) has just LEFT the air and
                         music is playing again: the item behind it is the
                         one that was just spent, and whatever else is spent
                         ahead is best removed now, before the play head
                         reaches it and MA records a silent replay -- the
                         routine moment the lifecycle itself provides
    periodic             nothing above has happened for
                         ma_observer.PERIODIC_RECONCILE_S of playing: the
                         backstop for a MISSED event (an outage the observer
                         never saw, a rewind that arrived as nothing, a
                         break that went terminal with no boundary) so a
                         spent item can never wait indefinitely -- routine,
                         and it asks the permission like an attach

No timer, no thread, no startup task, no station change trigger, no loop:
the observer's own poll is the only clock, and `periodic` is simply the
observer noticing how long its own polls have gone without a request. A
pass that reports more_remaining leaves the remainder to the armed chain
or to the next qualifying moment, which now always comes.

WHAT THIS MODULE NEVER DOES. It does not classify a queue item, pre-select
a candidate, read the queue, or touch Music Assistant; it calls
queue_hygiene.run_pass with the default engine limits and reports what came
back. It does not consult MUSICBRAIN_QUEUE_HYGIENE_CONTROL (that switch
exists for the manual POST route only) and it does not call that route.
It issues no playback command of any kind: a regression is a reason to
look for stale program items, never a reason to correct the player.

THE GATE. One process-wide non-blocking lock. A request that arrives while
an automatic pass is running is dropped with a logged reason -- never
queued, never retried -- because the engine's own station lock and
minimum gap are the authority on mutation cadence and a second request a
moment later has nothing new to say. The observer runs its requests on
its own polling thread, in order, so the two triggers cannot overlap
each other there; the gate is what keeps that true from every caller.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Any

import queue_hygiene

logger = logging.getLogger("musicbrain")

TRIGGER_INITIAL_ATTACH = "initial_attach"
TRIGGER_POSITION_REGRESSED = "position_regressed"
TRIGGER_QUIET_RECOVERY = "quiet_recovery"
TRIGGER_PROGRAM_BOUNDARY = "program_boundary"
TRIGGER_PERIODIC = "periodic"
TRIGGERS = (TRIGGER_INITIAL_ATTACH, TRIGGER_POSITION_REGRESSED,
            TRIGGER_QUIET_RECOVERY, TRIGGER_PROGRAM_BOUNDARY,
            TRIGGER_PERIODIC)

#: How a pass ended, for the question an operator asks first: did the last
#: cleanup actually happen, and if not, is anything owed a retry?
#: `completed` is a pass that ran to its end (every candidate decided),
#: `bounded` one that stopped at its own cap or time budget with work done
#: and a remainder, `no_candidates` the clean case and `classified` a dry
#: run. A PERSISTENT refusal needs a person (the setting is off, the latch
#: is tripped): the periodic backstop slows down for it. Everything else is
#: momentary -- the lock, the gap, the player, the position, a read, an
#: abort on a moving head -- and the next moment simply asks again.
SUCCESS_OUTCOMES = frozenset({"completed", "bounded", "no_candidates", "classified"})
PERSISTENT_REFUSALS = frozenset({"disabled", "latched"})

#: THE TRIGGERS THAT DO NOT ASK THE PERMISSION SETTING.
#
# playback.queue_hygiene is a ROUTINE CLEANUP POLICY: may Parlorwave tidy
# its own spent programming out of a queue's future as ordinary
# housekeeping? An operator may reasonably say no.
#
# A confirmed backward boundary is a different question. The queue has
# moved back over items the station ALREADY AIRED, so those spent breaks
# and bulletins are future items again. Left alone they re-air, and --
# worse, because it needs no playback at all -- the scheduler can read one
# as the next MUSIC track and hand Smart/floor `next = "DJ break"`. That
# is a lifecycle fault in Parlorwave's OWN bookkeeping about its OWN
# items, not a tidiness preference, so reconciliation is owed regardless
# of the setting. Otherwise a long regression meets `queue_hygiene_off`
# and reconciles nothing, leaving every spent program item ahead of the
# head.
#
# The bypass is of the PERMISSION GATE ONLY. Every safety gate the engine
# owns still applies -- latch, authority, the station mutation lock, the
# refill / Start Radio exclusions, the minimum gap, the playing state, the
# buffer, positive ownership, positive staleness, the serving belt and the
# protected-sequence check -- so a mandatory pass still removes only an
# item it positively owns and positively knows to be spent, and still
# cannot touch one note of the customer's music.
#
# initial_attach is deliberately NOT here. A restart is not evidence that
# anything went backward; it is the routine moment, and it keeps asking.
# quiet_recovery is not here for the same reason: resuming at the SAME
# position is evidence of nothing but time passing, so it asks too. (A
# resume EARLIER than the last live position is a position_regressed.)
MANDATORY = (TRIGGER_POSITION_REGRESSED,)

# --- the bounded reconciliation chain ----------------------------------------
#
# WHY A CHAIN EXISTS AT ALL. One regression is one event, but Music
# Assistant does not deliver a rewind as one event. One rewind can
# arrive as TWO regressions a few seconds apart:
#
#   N -> N-3     mandatory, many candidates, a few deleted, ABORTED head_changed
#   N-3 -> M     mandatory, refused min_gap
#
# Both refusals are correct. The first pass stops because the head is
# still moving, which is exactly what it must do; but it has already
# stamped the gap on its first delete, so the second pass -- the one that
# sees the settled queue and the whole candidate set -- is refused by the
# storm belt, and most stale items stay ahead. Separately, a rewind with
# more candidates than one pass may delete removes 25 cleanly and leaves
# the rest with no continuation.
#
# So the gap is doing its job (it protects against a storm of INDEPENDENT
# events) and the cleanup still does not converge, because one event
# cannot finish its own work. The chain is the smallest thing that fixes
# that: a generation-tagged, capped sequence of passes belonging to ONE
# regression, each of which re-reads the queue and re-proves everything.
#
# ROUTINE PASSES CHAIN TOO. The attach pass (the one routine trigger) hits
# the same 25-delete bound: a restart's attach pass can remove 25 and leave
# the rest with "no continuation scheduled", or abort
# head_changed_after_delete leaving all of them. A chain armed by a
# routine pass KEEPS the routine policy -- every continuation still asks the
# playback.queue_hygiene permission -- so turning hygiene off stops it at
# the next pass. And a routine event never displaces a regression's chain.
#
# WHAT IT IS NOT. Not a loop, not recursion, not a scheduler, not a timer,
# not a thread. Nothing here deletes; it decides WHEN to ask the engine
# again. Every pass is a fresh, independent, fully-gated run_pass, so a
# candidate proven stale in pass 1 is proven again in pass 2 or not
# removed. The only gate a chained pass waives is the minimum gap, and
# only because the chain is the same event finishing, not a new one.

#: Consecutive observer polls that must report the SAME head before a
#: chained pass may run. The observer polls about every 5 s, so two polls
#: is ~5-10 s of a settled queue -- long enough that a rewind in flight
#: has landed, short enough that reconciliation still happens promptly.
#: A pass that runs while the head is moving aborts anyway; this is what
#: stops the chain from burning its budget discovering that.
STABILIZE_POLLS = 2

#: Backstop on work passes in ONE chain, counting the initial event pass.
#: It was 6 (150 deletions), sized for a modest backlog -- but a backlog
#: can reach hundreds of spent items ahead of the head, and one event
#: then cleared 150 and abandoned the rest. A chain whose passes
#: keep removing what is positively stale is ONE episode finishing its own
#: work, and it runs until nothing safely removable is left; each pass is
#: still bounded (MAX_DELETES_PER_PASS, the time budget, the settled-head
#: polls between passes). This is only the backstop against a chain that
#: never finishes: 3000 deletions, several times any plausible backlog.
#: A chain that reaches it stops and says so; the next qualifying event
#: starts a fresh one.
MAX_CHAIN_PASSES = 120

# --- an owed cleanup is delayed, never silently erased --------------------------
#
# A chain can be dropped by conditions that say nothing about the queue:
# the player stops mid-pass, or a second 1-step regression runs its own
# pass, meets the 60 s gap and stops the chain the first had armed (spent
# items left behind) -- the very sequence the chain exists for. The
# invariant:
#
#   Once a chain is armed for a queue, a MOMENTARY result delays it; only
#   completion, a real finding, the queue changing, a newer regression or
#   one of the two budgets below ends it.
#
# MOMENTARY = the pass deleted nothing, and the engine's own words say why
# it could not start: the player is not playing, the station mutation lock
# is busy, a refill claim is pending or Start Radio is in flight, Music
# Assistant reported no usable position (POSITION_UNAVAILABLE), or (only
# when a chain for the same queue is ALREADY armed) the minimum gap. Every
# other answer keeps its meaning: a delete Music Assistant refuses, a latch,
# a serving-gate disagreement, lost authority, an unreadable store or queue,
# an unknown outcome -- all still end the chain. With no chain armed the gap
# still refuses an independent event, exactly as before.
#
# THE TWO BUDGETS. Deletions cannot loop -- each one removes a positively
# stale item and nothing here creates candidates -- so what needs a bound is
# the number of engine calls. MAX_CHAIN_PASSES counts only passes that did
# work (deleted something, or were refused for anything but a momentary
# reason); MAX_IDLE_CONTINUATIONS counts consecutive momentary continuations
# and any work pass resets it. A momentary result schedules nothing: the
# observer's settled-PLAYING polls stay the only clock.
MAX_IDLE_CONTINUATIONS = 3

#: Consecutive WORK passes that removed nothing. With MAX_CHAIN_PASSES now a
#: backstop rather than a budget, this is what bounds a chain whose passes
#: keep finding a reason to go on -- a head that moves mid-pass, a time
#: budget spent before the first delete on a slow Music Assistant -- while
#: removing nothing: it is not converging, so it stops (no_progress_cap)
#: and costs at most this many full reads. Any pass that removes something
#: resets it.
MAX_STALLED_PASSES = 3

#: Exclusion reasons the engine gives for a condition that passes on its
#: own. Exact words: anything else it excludes for (an unreadable station
#: store) is a finding, not a pause.
MOMENTARY_EXCLUSIONS = frozenset({"refill_pending", "start_radio_in_flight"})

#: The engine's words for "Music Assistant reported no usable position" --
#: no buffer index, an index that does not anchor the head, no head at all.
#: An ended or restarting queue does this for a while (MA publishes the
#: index after the item, or none while ended). Nothing was deleted on a
#: guess, and the same pass will run once the position is back, so this is
#: a pause in the episode, never its end.
POSITION_UNAVAILABLE = frozenset({"no_index_in_buffer", "index_units_mismatch",
                                  "no_head_identity",
                                  # spent items held only by a buffer index
                                  # left over from an ended Flow
                                  "stale_buffer_index"})

#: Why a chain was armed.
CHAIN_HEAD_CHANGED = "head_changed"      # a pass aborted on a moving head
CHAIN_COUNT_LIMIT = "count_limit"        # a clean pass hit MAX_DELETES_PER_PASS
CHAIN_NOT_PLAYING = "player_not_playing" # the player stopped: cleanup owed
CHAIN_LOCK_BUSY = "lock_busy"            # the station mutation lock was held
CHAIN_EXCLUDED = "refill_or_radio_in_flight"
CHAIN_MIN_GAP = "min_gap"                # a new event met the gap
CHAIN_POSITION_UNAVAILABLE = "position_unavailable"   # MA reported no position

# CLEANUP OWED WHILE THE PLAYER IS STOPPED. A long rewind can leave more
# spent items than a few chained passes remove; if a cast group goes
# silent in the middle of a pass (`unsupported_state_after_delete`) and
# the chain stops as if the engine had refused it, then when playback
# comes back hours later with no new qualifying event, the remaining spent
# items play out as tombstones -- each one also costing the DJ that
# boundary. A player that
# is not playing is not an obstruction to the cleanup, only a pause in it:
# the chain stays armed (in memory, like every chain) and maybe_continue's
# own settled-PLAYING rule decides when the next fully-gated pass runs.
# Nothing else changes -- no timer, no new trigger, no new deletion rule.

#: How a pass came to run, for telemetry.
KIND_EVENT = "initial_regression"
KIND_ATTACH = "initial_attach"
KIND_STABILIZE = "stabilization_retry"
KIND_CONTINUE = "count_limit_continuation"
KIND_RESUME = "resumed_after_stop"      # owed while the player was stopped

#: Why a chain stopped.
STOP_CLEAN = "no_candidates_remain"
STOP_CAP = "chain_pass_cap"
STOP_SUPERSEDED = "superseded_by_newer_regression"
STOP_REFUSED = "gate_refused"
STOP_LATCHED = "latch_tripped"
STOP_QUEUE_CHANGED = "queue_changed"
STOP_NO_PROGRESS = "no_progress_cap"

# Request outcomes that are THIS module's, distinct from the engine's own
# outcome vocabulary (disabled / latched / refused / lock_busy / gap /
# bounded / clean / aborted / error ...), which is passed through unchanged.
OUTCOME_SKIPPED = "skipped"
REASON_IN_FLIGHT = "auto_pass_in_flight"
REASON_NO_QUEUE = "no_queue_id"

_gate = threading.Lock()
_counts: Counter = Counter()
_last: dict[str, Any] | None = None
_last_success: dict[str, Any] | None = None   # the last pass that ran to its end
_last_refusal: dict[str, Any] | None = None   # the last answer that was not one
_state_lock = threading.Lock()

#: THE CHAIN. At most one, process-wide, guarded by _state_lock. A plain
#: dict rather than an object so `None` unambiguously means "no chain",
#: and so a process restart loses it harmlessly -- a dropped chain costs a
#: slower cleanup, never a wrong one, and the next regression starts a
#: fresh generation anyway. Never persisted for exactly that reason.
_chain: dict[str, Any] | None = None
_generation = 0
#: (queue_id, generation, thread) of the chained pass maybe_continue is
#: running right now, else None: the ONLY pass the chain's recovery
#: evidence is lent to (get_recovery_evidence).
_continuing: tuple[str, int, int] | None = None


def _reset() -> None:
    """Test seam: forget the bounded diagnostics (never touches the gate)."""
    global _last, _chain, _generation, _continuing
    with _state_lock:
        _counts.clear()
        _last = None
        _chain = None
        _generation = 0
        _continuing = None
    global _last_success, _last_refusal
    _last_success = _last_refusal = None


def chain_status() -> dict[str, Any] | None:
    """The live chain, or None. Counts and words only."""
    with _state_lock:
        return dict(_chain) if _chain is not None else None


def get_recovery_evidence(queue_id: str) -> dict[str, Any] | None:
    """The recovery evidence the armed chain for this queue was given -- the
    proof its event pass VALIDATED (the engine reports it) -- and ONLY to
    that chain's own continuation pass, while maybe_continue runs it on
    this thread; anyone else asking gets None. A diagnostic read: the
    engine consults no module for evidence at all --
    maybe_continue hands the chain's evidence to its own run_pass call
    explicitly -- so any other pass on the queue (a new event, an attach,
    the manual route, a dry run) brings its own or has none. Recovery-only
    authority lasts exactly as long as the episode it was proven for, and
    is re-checked with ownership on each of its passes. (Until the engine
    reported it, chains were armed with nothing and every continuation of
    a recovery-only pass was refused not_authoritative.)"""
    with _state_lock:
        if (_chain and _chain.get("queue_id") == queue_id
                and _continuing == (queue_id, _chain["generation"],
                                    threading.get_ident())):
            ev = _chain.get("recovery_evidence")
            return dict(ev) if ev else None
    return None


def _arm_chain(queue_id: str, reason: str, passes: int,
               generation: int, *, mandatory: bool = True,
               trigger: str = TRIGGER_POSITION_REGRESSED,
               recovery_evidence: dict[str, Any] | None = None) -> None:
    """Arm (or re-arm) the chain for THIS generation. A chain armed by an
    older generation is simply overwritten: the newer regression describes
    the queue as it is now, and the older one's remaining work is a
    subset of what the newer pass will find for itself.

    `mandatory` is the policy of the event that armed it, and every
    continuation runs under that same policy."""
    global _chain
    with _state_lock:
        _chain = {"generation": generation, "queue_id": queue_id,
                  "reason": reason, "passes": passes, "idle": 0,
                  "mandatory": mandatory, "trigger": trigger,
                  "last_head": None, "stable_polls": 0, "stopped": None,
                  "recovery_evidence": dict(recovery_evidence) if recovery_evidence else None}


def _chain_after_routine(queue_id: str, trigger: str,
                         result: dict[str, Any],
                         recovery_evidence: dict[str, Any] | None = None) -> int | None:
    """Arm, keep or close the chain after a ROUTINE pass. Returns the
    generation armed, or None.

    A regression's chain on this queue is never displaced by a routine
    event: it waives the permission for positive-evidence reasons a
    routine pass does not have, and its own passes will find whatever the
    routine pass left. Otherwise a routine remainder arms a routine chain
    under a fresh generation (so a later regression supersedes it), and a
    routine pass that finished closes a routine chain it finished."""
    global _generation
    more, why = _chain_verdict(result)
    moment = _momentary(result)
    if moment == CHAIN_MIN_GAP:
        more, why = False, STOP_REFUSED      # the gap refuses a NEW cleanup
    with _state_lock:
        cur = _chain
        if (cur is not None and cur.get("mandatory", True)
                and cur["queue_id"] == queue_id):
            return None
        if not more:
            stop = (cur is not None and cur["queue_id"] == queue_id)
            gen = None
        else:
            _generation += 1
            gen = _generation
            stop = False
    if stop:
        _stop_chain(why)
        return None
    if gen is None:
        return None
    _arm_chain(queue_id, why, passes=0 if moment else 1, generation=gen,
               mandatory=False, trigger=trigger,
               recovery_evidence=recovery_evidence)
    logger.info("queue_hygiene_chain_armed gen=%s queue=%s reason=%s "
                "after=%s policy=routine", gen, queue_id, why,
                result.get("outcome"))
    return gen


def _stop_chain(why: str) -> None:
    global _chain
    with _state_lock:
        if _chain is not None:
            _chain = {**_chain, "stopped": why}
            logger.info("queue_hygiene_chain_stopped gen=%s queue=%s "
                        "passes=%d reason=%s", _chain["generation"],
                        _chain["queue_id"], _chain["passes"], why)
            _chain = None


def _player_stopped(result: dict[str, Any]) -> bool:
    """Did the engine stop ONLY because the player is not in a supported
    playing state -- at the snapshot (`unsupported_state`) or on a head
    re-check before or after a delete (`aborted`, `unsupported_state...`)?
    The engine's own words, nothing inferred."""
    outcome = str(result.get("outcome") or "")
    reason = str(result.get("reason") or "")
    if outcome == "unsupported_state":
        return True
    return outcome == "aborted" and (
        reason.startswith("unsupported_state:")
        or reason.startswith("unsupported_state_after_delete:"))


def _position_unavailable(result: dict[str, Any]) -> bool:
    """Did the engine stop ONLY because Music Assistant reported no usable
    position -- refusing at the snapshot, or aborting on a re-check before
    or after a delete? The engine's own words, nothing inferred."""
    outcome = str(result.get("outcome") or "")
    reason = str(result.get("reason") or "")
    if outcome in ("refused", "unreadable"):
        return reason in POSITION_UNAVAILABLE
    word = reason.split(":", 1)[0]
    return outcome == "aborted" and (
        word in POSITION_UNAVAILABLE
        or word.removesuffix("_after_delete") in POSITION_UNAVAILABLE)


def _momentary(result: dict[str, Any]) -> str | None:
    """Why this pass did NO work only because of a condition that passes on
    its own, or None. The engine's own words; a pass that deleted anything,
    or tripped the latch, is never momentary."""
    if result.get("latch_tripped") or result.get("deleted"):
        return None
    outcome = str(result.get("outcome") or "")
    reason = str(result.get("reason") or "")
    if _player_stopped(result):
        return CHAIN_NOT_PLAYING
    if _position_unavailable(result):
        return CHAIN_POSITION_UNAVAILABLE
    if outcome == "lock_busy":
        return CHAIN_LOCK_BUSY
    if outcome == "excluded" and reason in MOMENTARY_EXCLUSIONS:
        return CHAIN_EXCLUDED
    if outcome == "min_gap":
        return CHAIN_MIN_GAP
    return None


def _keep_for_event(queue_id: str, *, mandatory: bool,
                    recovery_evidence: dict[str, Any] | None = None) -> int | None:
    """A new event's pass was momentary: an ARMED chain for the same queue
    stays exactly as it is -- its generation, its budget, its streak --
    except that a mandatory event upgrades it (never the reverse). Returns
    that chain's generation, or None when no chain for this queue exists."""
    with _state_lock:
        cur = _chain
        if cur is None or cur["queue_id"] != queue_id:
            return None
        if mandatory:
            cur["mandatory"] = True
        if recovery_evidence and not cur.get("recovery_evidence"):
            cur["recovery_evidence"] = dict(recovery_evidence)
        return cur["generation"]


def _chain_verdict(result: dict[str, Any]) -> tuple[bool, str]:
    """Should the chain continue after this pass, and if not, why not?

    Reads only the engine's own result vocabulary. Anything that is not a
    recognised 'there is more to do' shape stops the chain, so a new
    engine outcome defaults to stopping rather than to looping."""
    outcome = str(result.get("outcome") or "")
    if result.get("latch_tripped"):
        return False, STOP_LATCHED
    moment = _momentary(result)
    if moment is not None:
        return True, moment                      # delayed, not erased
    if _player_stopped(result):
        return True, CHAIN_NOT_PLAYING           # owed until it plays again
    if _position_unavailable(result):
        return True, CHAIN_POSITION_UNAVAILABLE  # owed until MA says where
    if outcome == "aborted" and str(result.get("reason") or "").startswith(
            "head_changed"):
        return True, CHAIN_HEAD_CHANGED          # settle, then try again
    if outcome == "bounded" and result.get("more_remaining"):
        return True, CHAIN_COUNT_LIMIT           # budget, not obstruction
    if outcome in ("completed", "no_candidates", "clean"):
        if result.get("stale_buffer_held"):
            # removed what it could; what MA's stale buffer index held is
            # still owed (queue_hygiene.REASON_STALE_BUFFER)
            return True, CHAIN_POSITION_UNAVAILABLE
        return False, STOP_CLEAN
    if outcome in ("latched",):
        return False, STOP_LATCHED
    if outcome in ("refused", "excluded", "lock_busy", "disabled",
                   "unreadable", "error", "aborted"):
        return False, STOP_REFUSED
    return False, STOP_REFUSED


def _latch_word(result: dict[str, Any]) -> str:
    if result.get("latch_tripped"):
        return "tripped"
    if (result.get("preconditions") or {}).get("latch"):
        return "latched"
    return "clear"


def _summary(trigger: str, queue_id: str, steps: int | None,
             also: str | None, result: dict[str, Any] | None,
             outcome: str, reason: str, now: float, *,
             mandatory: bool = False,
             from_index: int | None = None,
             to_index: int | None = None) -> dict[str, Any]:
    """The whole record of one request, in counts and words. A reader must
    be able to answer "what did the rewind do, and what did we do about
    it" from this alone -- never from raw Music Assistant state."""
    out: dict[str, Any] = {
        "trigger": trigger, "also": also, "queue_id": queue_id,
        "steps": steps, "at": now, "outcome": outcome, "reason": reason,
        # where the queue went
        "from_index": from_index, "to_index": to_index,
        # what we found, and what we did about it
        "mandatory": mandatory, "policy": None, "permission_bypassed": False,
        "gap_waived": False, "chain_generation": None, "chain_pass": None,
        "chain_kind": None,
        "stale_found": 0, "stale_removed": 0,
        "deleted": 0, "remaining": 0, "more_remaining": False,
        "candidates": 0, "preserved": {}, "preserved_total": 0,
        "ordinary_authority": "", "recovery_only_authority": False,
        "recovery_authoritative": False, "recovery_reason": "",
        "retained_reasons": {},
        "latch": None, "pass_id": None,
        "lock_held_s": 0.0, "elapsed_s": 0.0,
    }
    if result is not None:
        deleted = len(result.get("deleted") or ())
        candidates = int(result.get("candidates") or 0)
        preserved = dict(result.get("preserved") or {})
        out.update({
            "policy": result.get("policy"),
            "permission_bypassed": bool(result.get("permission_bypassed")),
            "gap_waived": bool(result.get("gap_waived")),
            "deleted": deleted,
            # Named for the question a rewind asks, not for the engine's
            # internals: how many spent owned items were ahead, and how
            # many of them did this bounded pass actually remove.
            "stale_found": candidates,
            "stale_removed": deleted,
            "remaining": int(result.get("remaining") or 0),
            "more_remaining": bool(result.get("more_remaining")),
            "candidates": candidates,
            "preserved": preserved,
            "preserved_total": int(result.get("protected_count") or 0),
            "ordinary_authority": result.get("ordinary_authority") or "",
            "recovery_only_authority": bool(result.get("recovery_only_authority")),
            "recovery_authoritative": bool(result.get("recovery_authoritative")),
            "recovery_reason": result.get("recovery_reason") or "",
            "retained_reasons": dict(result.get("retained_reasons") or {}),
            "latch": _latch_word(result),
            "pass_id": (result.get("pass_id") or "")[:8] or None,
            "lock_held_s": result.get("lock_held_s", 0.0),
            "elapsed_s": result.get("elapsed_s", 0.0),
        })
    return out


def retry_class(outcome: str | None, reason: str | None = None) -> str | None:
    """'success', 'persistent' (needs a person) or 'momentary' (the next
    moment asks again); None for an unknown outcome word."""
    if not outcome:
        return None
    if outcome in SUCCESS_OUTCOMES:
        return "success"
    if outcome in PERSISTENT_REFUSALS:
        return "persistent"
    return "momentary"


def _record(summary: dict[str, Any]) -> None:
    global _last
    with _state_lock:
        _counts[summary["trigger"]] += 1
        _counts["outcome:" + str(summary["outcome"])] += 1
        _last = dict(summary)
    # The two answers a status page needs beside the last request: when
    # did a cleanup last run to its end, and what was the last refusal
    # (with whether a person is needed or the next moment retries it).
    global _last_success, _last_refusal
    cls = retry_class(summary.get("outcome"), summary.get("reason"))
    brief = {k: summary.get(k) for k in ("at", "trigger", "outcome", "reason",
                                          "deleted", "candidates", "remaining",
                                          "more_remaining", "chain_kind")}
    brief["retry"] = cls
    with _state_lock:
        if cls == "success":
            _last_success = brief
        elif cls in ("persistent", "momentary"):
            _last_refusal = brief


def is_mandatory(trigger: str, also: str | None = None) -> bool:
    """Does this cycle carry positive evidence of a regression? True when
    either the trigger or the coalesced second trigger is one, so the
    combined `initial_attach also=position_regressed` cycle -- a restart
    that came back further back than it left -- reconciles too."""
    return trigger in MANDATORY or (also or "") in MANDATORY


def request(trigger: str, queue_id: str, *, steps: int | None = None,
            also: str | None = None, from_index: int | None = None,
            to_index: int | None = None,
            recovery_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask the hygiene engine for ONE bounded pass over `queue_id`, now, because
    `trigger` happened. Returns a bounded summary (counts and words, never
    queue item ids or tokens) and logs one structured line.

    `steps` is the regression magnitude when the trigger is a regression
    (reported, never acted on: the engine does not care whether the queue
    went back three positions or three hundred). `also` names a second
    trigger that coincided in the same observer cycle and was coalesced
    into this one request.

    `recovery_evidence` is the OBSERVER'S proof of a post-quiet recovery
    event (ma_observer._request_hygiene builds it from the quiet-stretch
    verdicts it saw), handed to the engine for this one pass and nothing
    else; the engine judges it. The evidence the pass VALIDATED comes back
    in the result and arms the chain, so the episode's continuations carry
    their own proof. No module holds it in between: a caller that brings
    none -- a dry run, the manual route, an attach -- gets none.

    Operational refusals from the engine (disabled, latched, not ours, lock
    busy, gap) come back as the engine's own outcome and are final for this
    event: no retry, no wait, no rescheduling. A programming defect inside
    the engine propagates, as queue_hygiene promises; the observer fences
    its call."""
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown hygiene trigger: {trigger!r}")
    now = time.time()
    mandatory = is_mandatory(trigger, also)
    queue_id = (queue_id or "").strip()
    if not queue_id:
        summary = _summary(trigger, "", steps, also, None,
                           OUTCOME_SKIPPED, REASON_NO_QUEUE, now,
                           mandatory=mandatory, from_index=from_index,
                           to_index=to_index)
        _record(summary)
        return summary
    if not _gate.acquire(blocking=False):
        summary = _summary(trigger, queue_id, steps, also, None,
                           OUTCOME_SKIPPED, REASON_IN_FLIGHT, now,
                           mandatory=mandatory, from_index=from_index,
                           to_index=to_index)
        _record(summary)
        logger.info("queue_hygiene_auto trigger=%s queue=%s steps=%s "
                    "outcome=%s reason=%s (dropped, not queued)",
                    trigger, queue_id, steps, OUTCOME_SKIPPED,
                    REASON_IN_FLIGHT)
        return summary
    global _generation
    if mandatory:
        # A new regression is a new GENERATION: any chain still pending
        # from an older one becomes inert here, before the pass runs, so
        # two rewinds can never interleave their cleanups.
        with _state_lock:
            _generation += 1
            gen = _generation
    else:
        gen = None
    try:
        # The one automatic call into the engine: default limits (the
        # engine's reviewed bounds), default dependencies (the real queue, the
        # real store, the real lock), no dry run. `mandatory` waives the
        # permission setting ONLY, and only on positive regression
        # evidence; the engine keeps every other gate. The FIRST pass of a
        # chain never waives the gap -- it is an ordinary independent
        # event and the storm belt applies to it exactly as before.
        result = queue_hygiene.run_pass(queue_id, mandatory=mandatory,
                                        recovery_evidence=recovery_evidence)
    finally:
        _gate.release()
    moment = _momentary(result)
    kept = (_keep_for_event(queue_id, mandatory=mandatory,
                            recovery_evidence=result.get("recovery_evidence"))
            if moment is not None else None)
    if kept is not None:
        # The obligation already exists: this event delays it, no more.
        gen = armed = kept
        logger.info("queue_hygiene_chain_kept gen=%s queue=%s reason=%s "
                    "trigger=%s", kept, queue_id, moment, trigger)
    elif mandatory:
        more, why = _chain_verdict(result)
        if moment == CHAIN_MIN_GAP:
            more, why = False, STOP_REFUSED  # the gap refuses a NEW cleanup
        if more:
            _arm_chain(queue_id, why, passes=0 if moment else 1,
                       generation=gen,
                       recovery_evidence=result.get("recovery_evidence"))
            logger.info("queue_hygiene_chain_armed gen=%s queue=%s "
                        "reason=%s after=%s", gen, queue_id, why,
                        result.get("outcome"))
        else:
            _stop_chain(why)
        armed = gen if more else None
    else:
        gen = armed = _chain_after_routine(queue_id, trigger, result,
                                           recovery_evidence=result.get("recovery_evidence"))
    summary = _summary(trigger, queue_id, steps, also, result,
                       str(result.get("outcome") or ""),
                       str(result.get("reason") or ""), now,
                       mandatory=mandatory, from_index=from_index,
                       to_index=to_index)
    summary["chain_generation"] = gen
    summary["chain_pass"] = 1 if gen is not None else None
    summary["chain_kind"] = (KIND_EVENT if mandatory
                             else KIND_ATTACH if gen is not None else None)
    _record(summary)
    logger.info("queue_hygiene_auto trigger=%s%s queue=%s steps=%s "
                "policy=%s outcome=%s reason=%s deleted=%d candidates=%d "
                "remaining=%d more_remaining=%s latch=%s pass=%s "
                "lock_held_s=%.3f",
                trigger, f" also={also}" if also else "", queue_id, steps,
                summary["policy"], summary["outcome"], summary["reason"],
                summary["deleted"], summary["candidates"],
                summary["remaining"], summary["more_remaining"],
                summary["latch"], summary["pass_id"],
                summary["lock_held_s"])
    if mandatory:
        # THE REWIND RECORD. One line that answers the whole question a
        # morning report asks -- where the queue went, what was spent and
        # ahead, what was removed, what was kept -- without anyone opening
        # Music Assistant.
        logger.info(
            "queue_rewind_reconciled queue=%s from_index=%s to_index=%s "
            "steps=%s stale_found=%d stale_removed=%d remaining=%d "
            "preserved=%d preserved_by=%s permission_bypassed=%s "
            "outcome=%s reason=%s latch=%s",
            queue_id, summary["from_index"], summary["to_index"],
            summary["steps"], summary["stale_found"],
            summary["stale_removed"], summary["remaining"],
            summary["preserved_total"],
            summary["preserved"] or "{}",
            summary["permission_bypassed"], summary["outcome"],
            summary["reason"] or "-", summary["latch"])
    rec_ev = (result or {}).get("recovery_evidence") if isinstance(result, dict) else None
    if (trigger == TRIGGER_QUIET_RECOVERY
            or rec_ev is not None
            or summary.get("recovery_only_authority")
            or summary.get("recovery_authoritative")):
        player_id = ((rec_ev or {}).get("player_id")
                     or ((result or {}).get("snapshot") or {}).get("player_id")
                     or queue_id)
        logger.info(
            "queue_recovery_reconciled trigger=%s ordinary_authority=%s "
            "recovery_only_authority=%s queue=%s player=%s stale_candidates=%d "
            "removed=%d retained_reasons=%s outcome=%s reason=%s",
            trigger,
            "present" if summary.get("ordinary_authority") else "absent",
            "used" if summary.get("recovery_only_authority") else "not_used",
            queue_id, player_id,
            summary.get("stale_found", 0),
            summary.get("stale_removed", 0),
            summary.get("retained_reasons") or "{}",
            summary.get("outcome"),
            summary.get("reason") or "-")
    if summary["more_remaining"]:
        # Nothing here schedules, loops or waits. The remainder is either
        # the armed chain's (offered to later observer polls) or the next
        # qualifying event's.
        if armed is not None:
            logger.info("queue_hygiene_auto_remainder trigger=%s queue=%s "
                        "remaining=%d continues on settled observer polls "
                        "(chain gen=%s)", trigger, queue_id,
                        summary["remaining"], armed)
        else:
            logger.info("queue_hygiene_auto_remainder trigger=%s queue=%s "
                        "remaining=%d left for the next qualifying event "
                        "(no continuation scheduled)",
                        trigger, queue_id, summary["remaining"])
    return summary


def maybe_continue(queue_id: str, head_id: str | None, *,
                   playing: bool = True) -> dict[str, Any] | None:
    """Give an armed reconciliation chain ONE more bounded pass, if and
    only if the queue head has settled. Returns a summary, or None when
    nothing ran (the overwhelmingly common case).

    THE OBSERVER'S POLL IS THE CLOCK. Nothing here sleeps, schedules,
    spawns or retries by itself: the observer already looks at this queue
    every few seconds and simply offers each look to the chain. That is
    what makes this a state machine rather than a loop -- if the observer
    stops, the chain stops with it, and a process restart forgets it
    entirely, which costs a slower cleanup and nothing else.

    `head_id` is the observer's current head identity; this module still
    reads no queue and touches no Music Assistant. Head STABILITY is a
    precondition: a pass run while the head is moving aborts, and an aborted pass that has already deleted
    something has spent budget to learn what one comparison could have
    told it."""
    global _chain, _continuing
    with _state_lock:
        ch = dict(_chain) if _chain is not None else None
    if ch is None:
        return None
    if ch["queue_id"] != queue_id:
        # The station moved elsewhere; this chain describes a queue we are
        # no longer the authority for.
        _stop_chain(STOP_QUEUE_CHANGED)
        return None
    if not playing or not head_id:
        # Idle, paused, gone or an unknown head: not a settled queue, and
        # the engine would refuse anyway. Reset the streak, keep the chain.
        with _state_lock:
            if _chain is not None:
                _chain["stable_polls"] = 0
                _chain["last_head"] = None
        return None
    with _state_lock:
        if _chain is None:
            return None
        if _chain["last_head"] == head_id:
            _chain["stable_polls"] += 1
        else:
            _chain["last_head"] = head_id
            _chain["stable_polls"] = 1
        stable = _chain["stable_polls"]
        passes = _chain["passes"]
        gen = _chain["generation"]
        reason = _chain["reason"]
        mandatory = _chain.get("mandatory", True)
        trigger = _chain.get("trigger", TRIGGER_POSITION_REGRESSED)
        recovery_evidence = _chain.get("recovery_evidence")
    if stable < STABILIZE_POLLS:
        return None                      # still settling
    if passes >= MAX_CHAIN_PASSES:
        _stop_chain(STOP_CAP)
        return None
    if not _gate.acquire(blocking=False):
        return None                      # a request is running; try next poll
    now = time.time()
    try:
        # The chained pass, under the policy of the event that armed the
        # chain: a regression's chain stays mandatory, an attach's stays
        # routine and keeps asking the permission. `ignore_gap` because
        # this is the SAME event finishing its own work -- the storm belt
        # still applies to every independent event. Every other gate is
        # asked. The chain's recovery evidence is lent to THIS pass only.
        with _state_lock:
            _continuing = (queue_id, gen, threading.get_ident())
        # The chain's own validated evidence rides THIS call and no other:
        # the engine consults no module for it, so a pass that did not
        # bring it cannot borrow it.
        result = queue_hygiene.run_pass(queue_id, mandatory=mandatory,
                                        ignore_gap=True,
                                        recovery_evidence=recovery_evidence)
    finally:
        with _state_lock:
            _continuing = None
        _gate.release()
    # Re-check the generation: a newer regression may have superseded this
    # chain while the pass was running. Its work is still valid (it was a
    # fully-gated pass) but it must not extend a chain that is gone.
    # A momentary pass spends nothing from the work budget, only from the
    # no-work streak; a pass that did any work spends one and ends the
    # streak (including one that deleted and THEN met a stopped player).
    moment = _momentary(result)
    idle = stalled = 0
    with _state_lock:
        superseded = (_chain is None or _chain["generation"] != gen)
        if not superseded:
            if moment is None:
                _chain["passes"] = passes + 1
                _chain["idle"] = 0
                _chain["stalled"] = (0 if result.get("deleted")
                                     else _chain.get("stalled", 0) + 1)
            else:
                _chain["idle"] = _chain.get("idle", 0) + 1
            idle = _chain["idle"]
            stalled = _chain.get("stalled", 0)
            _chain["stable_polls"] = 0
    work_pass = passes + (0 if moment else 1)
    kind = (KIND_STABILIZE if reason == CHAIN_HEAD_CHANGED
            else KIND_RESUME if reason == CHAIN_NOT_PLAYING
            else KIND_CONTINUE)
    summary = _summary(trigger, queue_id, None, None,
                       result, str(result.get("outcome") or ""),
                       str(result.get("reason") or ""), now,
                       mandatory=mandatory)
    summary.update(chain_generation=gen, chain_pass=work_pass,
                   chain_kind=kind, chain_idle=idle, momentary=moment)
    _record(summary)
    logger.info("queue_hygiene_chain gen=%s pass=%d idle=%d kind=%s queue=%s "
                "outcome=%s reason=%s deleted=%d candidates=%d remaining=%d "
                "gap_waived=%s latch=%s lock_held_s=%.3f",
                gen, work_pass, idle, kind, queue_id, summary["outcome"],
                summary["reason"] or "-", summary["deleted"],
                summary["candidates"], summary["remaining"],
                summary["gap_waived"], summary["latch"],
                summary["lock_held_s"])
    rec_ev = (result or {}).get("recovery_evidence") if isinstance(result, dict) else None
    if (trigger == TRIGGER_QUIET_RECOVERY
            or rec_ev is not None
            or summary.get("recovery_only_authority")
            or summary.get("recovery_authoritative")):
        player_id = ((rec_ev or {}).get("player_id")
                     or ((result or {}).get("snapshot") or {}).get("player_id")
                     or queue_id)
        logger.info(
            "queue_recovery_reconciled trigger=%s ordinary_authority=%s "
            "recovery_only_authority=%s queue=%s player=%s stale_candidates=%d "
            "removed=%d retained_reasons=%s outcome=%s reason=%s",
            trigger,
            "present" if summary.get("ordinary_authority") else "absent",
            "used" if summary.get("recovery_only_authority") else "not_used",
            queue_id, player_id,
            summary.get("stale_found", 0),
            summary.get("stale_removed", 0),
            summary.get("retained_reasons") or "{}",
            summary.get("outcome"),
            summary.get("reason") or "-")
    if superseded:
        _stop_chain(STOP_SUPERSEDED)
        return summary
    if moment is not None and idle >= MAX_IDLE_CONTINUATIONS:
        _stop_chain(STOP_NO_PROGRESS)            # persistent, not momentary
        return summary
    more, why = _chain_verdict(result)
    if more and stalled >= MAX_STALLED_PASSES:
        more, why = False, STOP_NO_PROGRESS      # working, removing nothing
    if more:
        with _state_lock:
            if _chain is not None and _chain["generation"] == gen:
                _chain["reason"] = why
    else:
        _stop_chain(why)
    return summary


def status() -> dict[str, Any]:
    """Bounded diagnostics: which triggers exist, whether an automatic pass
    is running right now, how many requests each trigger has made, and the
    last request's summary. No ids, no paths."""
    with _state_lock:
        last = dict(_last) if _last is not None else None
        counts = dict(_counts)
        success = dict(_last_success) if _last_success is not None else None
        refusal = dict(_last_refusal) if _last_refusal is not None else None
    return {"triggers": list(TRIGGERS), "mandatory": list(MANDATORY),
            "in_flight": _gate.locked(),
            "chain": chain_status(),
            # observability for an unattended house: the last cleanup that
            # ran to its end, and the last refusal with its retry class
            "last_success": success, "last_refusal": refusal,
            "chain_limits": {"stabilize_polls": STABILIZE_POLLS,
                             "max_chain_passes": MAX_CHAIN_PASSES,
                             "max_idle_continuations":
                                 MAX_IDLE_CONTINUATIONS},
            "counts": counts, "last": last}
