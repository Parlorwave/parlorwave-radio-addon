#!/usr/bin/env python3
"""Ownership-aware removal of Parlorwave's OWN stale program items from the
future of a Music Assistant queue.

THE PROBLEM IT ANSWERS. A Music Assistant queue item is addressable forever,
and program audio is not: an aired DJ break or a spent news bulletin stays
queued as an item whose URL now answers a tombstone. On a station that runs
for days those items accumulate ahead of the play head (dozens of program
references a day), and a restart rewind or a
listener scrolling forward meets a run of short silences where the station
once spoke. program_tokens keeps the TRANSPORT safe; this module keeps the
future of the queue honest, by removing items that are provably ours and
provably over.

WHAT IT MAY TOUCH, STATED ONCE. Music Assistant owns playback, the music and
the provider's own queue semantics. Parlorwave may remove only an item it
can positively identify as its own finite program element AND positively
classify as stale. It never rebuilds, reorders, moves, clears, pauses,
skips or seeks; the single verb it has is ma_queue.delete_queue_item, by
exact queue_item_id string, one item at a time.

OWNERSHIP IS NOT A URL SHAPE. media_ref.programming_ref reads a kind and a
token out of a route, and that is where ownership STARTS, never where it
ends: the token must resolve in Parlorwave's own durable state -- a
dj_breaks row for THIS queue, a news_bulletins row, or a retired_program_
tokens fact of the matching kind. A route-shaped URL with a token nothing
here knows is UNKNOWN and is left exactly where it is; so is a malformed
token, a DJ row that belongs to another queue, and a ledger fact of the
wrong kind. Pre-ledger DJ tokens whose rows were reaped before the ledger
existed are unrecoverable and stay. Title, artist, display name and
"looks like ours" are never consulted, for anything.

STALE IS A LIFECYCLE FACT, THEN A SERVING FACT. Classification says stale
from the row state or the retirement ledger. Before every actual delete the
audio-serving gate that answers Music Assistant's media requests
(dj_engine.audio_available / news_bulletins.audio_available) is asked as an
independent belt: it must already answer a tombstone for the token. If the
lifecycle says stale but the gate would serve live audio, the two truths
have drifted and the PASS ABORTS -- the disagreement is the finding, and
deleting anyway would be hiding it.

TWO SWITCHES, BOTH REQUIRED. `playback.queue_hygiene` (settings) is the
operator's permission. The SAFETY LATCH (a `meta` row in the sessions
database, see LATCH_KEY) is the engine's own belief that mutation is still
safe; it is tripped by this module when a pass cannot prove the customer's
music survived in order, it survives restarts, and nothing clears it but an
explicit clear_latch. Tripping the latch never rewrites the setting.

THE PASS, in order (run_pass):

    permission -> latch -> queue authority -> station mutation lock
    (non-blocking) -> no refill batch pending, no Start Radio in flight ->
    minimum gap since the last mutating pass -> ONE complete snapshot
    (head identity, current_index, index_in_buffer, the whole future) ->
    classify -> PROTECTED = the customer's music and the provider's picks
    (every future item with no program route); WATCHED = every other
    future item we will not remove (live, unknown or buffered program
    items) -> candidates nearest the head first -> for each, while
    the count and time budgets hold: a cheap probe (queue_state plus a
    one-item page proving current_index is still the head's position),
    same head, no index regression, supported state; a small page around
    the candidate's last known position (one complete read if it drifted
    out of the window), candidate still strictly future and beyond the
    buffer and not the item Music Assistant names as next, ownership +
    staleness re-proven, serving gate says tombstone, the gap stamp (once
    per pass), the probe AGAIN as the last look (same head, still beyond
    the buffer, still not MA's next), ONE delete, probe + page again,
    candidate gone, not still MA's next, and head unchanged ->
    closing check over the WHOLE queue as re-read: every deleted id is
    absent, and the protected sequence is a SUBSEQUENCE of what is there
    (appends and insertions are fine, a missing or reordered id is not)
    -> on violation, trip the latch with the ids that broke it. A watched
    id that went missing is REPORTED in the result and the log, never
    latched: it is not the customer's music, and nothing of ours removes
    queue items except this pass.

A head change, an index regression, a refused delete, a transport error, a
serving-gate disagreement or a delete Music Assistant accepted while still
naming the item as its next ABORTS the pass -- no continuation across a
track advance, no retry of a refused item. A candidate that has vanished,
slid into the buffer or become the item MA names as next is SKIPPED, which
is not evidence of anything: MA can commit its next item to the player
while that item is still beyond index_in_buffer, so the buffer alone does
not say what the player holds (a deleted next item can be requested,
answer 404, and send playback idle early). Running
out of count or time is an ordinary bounded exit, and the result says how
many positively stale candidates were left unreached (`remaining`,
`more_remaining`) so a caller can decide to continue -- in a NEW pass,
under a new lock acquisition, never by holding this one longer. None of
those trip the latch; only a protected-sequence violation, or a closing
read that cannot be completed after items were removed, does.

WHAT IS CAUGHT AND WHAT IS NOT. Operational failures become structured,
fail-closed results: Music Assistant transport (MAQueueError), the sessions
database (sqlite3.Error, OSError) and lock contention (StationMutationBusy).
A programming defect -- an AttributeError, a TypeError, an assertion -- is
NOT swallowed: it propagates, with the lock released on the way out, so a
bug cannot hide behind "the pass returned an error". A future caller may
fence the call site; this module will not fence itself.

WHO CALLS IT. queue_hygiene_triggers.py is the one automatic caller: the
observer's first stable playing reconcile of an attach episode and a
confirmed position regression each ask for ONE pass with these default
limits; a pass that reports more_remaining leaves the
rest for the next such event. The manual POST route (app.py, behind
MUSICBRAIN_QUEUE_HYGIENE_CONTROL) is the only other caller. There is
still no periodic scan and no continuation of any kind.

NEVER HOLD A SQLITE TRANSACTION WHILE HOLDING THE STATION LOCK; every read
here opens and closes its own connection.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import autoplay_watch
import ma_queue
import media_ref
import program_audio
import program_tokens
import service_clock
import station_lock

logger = logging.getLogger("musicbrain")

# --- vocabulary ----------------------------------------------------------------

KIND_DJ = program_tokens.KIND_DJ
KIND_NEWS = program_tokens.KIND_NEWS

#: What classification concludes about one item's OWNERSHIP AND LIFECYCLE.
V_DELETE = "delete"        # positively ours, positively stale
V_PRESERVE = "preserve"    # ours and live, or not ours at all (music)
V_UNKNOWN = "unknown"      # route-shaped but unprovable: left alone

#: Where the item sits relative to the play head. Only "future" is eligible.
ZONE_FUTURE = "future"
ZONE_BUFFERED = "buffered"   # at or below index_in_buffer: the player owns it
ZONE_UNKNOWN = "unknown"     # buffer index not reported: nothing is eligible

#: What a dry run would do, and what a mutating pass tries to do.
ACT_WOULD_DELETE = "would_delete"
ACT_PRESERVE = "preserve"
ACT_LEAVE_UNKNOWN = "leave_unknown"
ACT_PROTECTED = "buffered_protected"

#: A STALE BUFFER INDEX (seen on a cast group). When a finite
#: Flow queue runs out, MA derives the cursor from the player's elapsed time
#: over the Flow log and drops it back to the Flow's first item, while
#: index_in_buffer -- reset only by a new load -- still names the last item
#: it streamed. Every already-aired item between the two then reads as
#: player-owned. Deleting one is still forbidden (the buffer is MA's word),
#: but calling that "no candidates" would end the episode and leave the
#: spent items owed to nobody. Spent program items held ONLY because the buffer
#: index reaches past the head's own successor are a position MA has not
#: finished reporting, so the pass says so and the cleanup stays owed
#: (queue_hygiene_triggers.POSITION_UNAVAILABLE); the next pass after the
#: re-open -- a new load, the buffer at the head again -- removes them. The
#: head's own successor is excluded: MA preloads it in ordinary playback.
REASON_STALE_BUFFER = "stale_buffer_index"


def stale_buffer_held(classified: list[dict[str, Any]],
                      current_position: int, index_in_buffer: Any) -> int:
    """How many spent program items are held only by a buffer index that
    reaches past the head's successor (0 in ordinary playback)."""
    if (not isinstance(index_in_buffer, int) or isinstance(index_in_buffer, bool)
            or index_in_buffer <= current_position + 1):
        return 0
    return sum(1 for r in classified
               if r["action"] == ACT_PROTECTED
               and r["position"] > current_position + 1)

#: WHY a pass was allowed to run at all -- the two policies, deliberately
#: distinct. A rewind is not a cleanup preference: it is a lifecycle fault
#: in which items the station has ALREADY AIRED are in the future again,
#: and leaving them there makes the product re-air spent programming and
#: hand the scheduler a break as "next music". So the optional setting
#: governs ROUTINE cleanup only, and a confirmed regression reconciles
#: ownership regardless of it. Every OTHER gate is identical in both
#: policies -- latch, authority, lock, exclusions, gap, playing state,
#: buffer, positive ownership, positive staleness, the serving belt and
#: the protected-sequence check. Mandatory buys exactly one thing: the
#: permission gate is not asked.
POLICY_ROUTINE = "routine"            # the operator's playback.queue_hygiene
POLICY_MANDATORY = "mandatory"        # a confirmed queue-position regression

#: Ownership bases (why we believe the token is ours).
OWN_DJ_ROW = "dj_row"
OWN_DJ_LEDGER = "dj_ledger"
OWN_NEWS_ROW = "news_row"
OWN_NEWS_LEDGER = "news_ledger"

#: DJ lifecycle vocabulary, restated from dj_engine so a change there is a
#: conscious change here (pinned by test).
DJ_TERMINAL_STATES = ("completed", "missed", "revoked", "failed")
DJ_LIVE_STATES = ("planned", "rendering", "ready", "queued", "airing")

NEWS_STALE_STATES = ("expired", "failed")
NEWS_LIVE_STATES = ("discovered", "downloading")

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")

#: Queue states in which a pass may mutate. Deliberately "playing" only:
#: the automatic triggers fire on a playing queue, index_in_buffer is
#: meaningful there, and anything else is a state this engine has no
#: evidence for.
SUPPORTED_STATES = ("playing",)

# --- bounds. Named, conservative ---------------------------------------------
#
# THE LOCK IS THE BUDGET. A pass holds the station mutation lock, and the
# two other holders are user-visible: Start Radio (station_radio) and a
# refill (station_refill) both acquire it with a 30 s timeout and REFUSE
# rather than write when it does not come free. So a pass may never hold it
# long enough for a person to feel it. Per attempted delete the pass makes
# nine small calls (queue_state, a one-item anchor page, a 17-item window,
# queue_state and anchor again as the last look, the delete, then
# queue_state, anchor and window again): on the LAN each is a few
# milliseconds, so 25 deletes are ~1-3 s of lock time plus one
# complete read at each end (tens of ms and a few MB per 500-item page).
# Five seconds covers that with room and stays far below the waiters'
# timeout; a stalled Music Assistant call is bounded by MA_TIMEOUT_S and
# surfaces as an abort. The backlog is cleared over several passes on
# purpose.
MAX_DELETES_PER_PASS = 25
PASS_TIME_BUDGET_S = 5.0
#: Belt against a trigger that fires repeatedly (a restart loop, a rewind
#: storm): two mutating passes are never closer than this, measured from
#: the moment a pass was about to attempt its first delete (stamped just
#: before that delete's last look), persisted across restarts.
#: One minute, not ten: this is a storm belt, not a schedule -- at most
#: MAX_DELETES_PER_PASS removals a minute -- and a bounded continuation
#: must be able to finish a backlog of a hundred items in minutes.
MIN_PASS_GAP_S = 60.0
#: Items read either side of a candidate's last known list position when
#: revalidating it. Positions only drift when something is inserted or
#: removed NEARER the head than the candidate (a NEXT insertion, a
#: listener's edit) -- rare within a pass, and a drift beyond the window
#: simply costs one complete read that refreshes every remaining hint.
LOCATE_WINDOW = 8
#: The station mutation lock is TRIED, never waited for. Hygiene is the
#: lowest-priority writer there is; a refill or Start Radio in flight simply
#: means "not now".
LOCK_WAIT_S = 0.0

#: Diagnostic lists persisted with a tripped latch are capped so a latch
#: record can never become a queue snapshot.
DIAG_ID_CAP = 50
#: Hard stop for the closing whole-queue read: defensive against a pager
#: that never returns a short page, and far above any realistic queue.
#: Hitting it makes the closing read
#: "unreadable", which trips the latch -- so it must never be reachable by
#: an ordinary queue.
WHOLE_QUEUE_CAP = 50_000

# --- persistent safety latch (sessions.db `meta`) --------------------------------

#: JSON: {tripped_at, reason, queue_id, pass_id, detail}. Absent = clear.
LATCH_KEY = "queue_hygiene:latch"
#: ISO stamp of the last pass that attempted a delete. Absent = never.
LAST_PASS_KEY = "queue_hygiene:last_pass_at"

#: Latch reasons.
LATCH_PROTECTED_VIOLATED = "protected_sequence_violated"
LATCH_CLOSING_UNREADABLE = "closing_read_unreadable"
LATCH_UNREADABLE = "latch_unreadable"   # synthetic: the row could not be read


@dataclass(frozen=True)
class Limits:
    max_deletes: int = MAX_DELETES_PER_PASS
    time_budget_s: float = PASS_TIME_BUDGET_S
    min_gap_s: float = MIN_PASS_GAP_S
    lock_wait_s: float = LOCK_WAIT_S


def _connect(db_path=None):
    import playback
    playback.ensure_sessions_schema(db_path)
    return playback._connect(db_path)


def _iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).replace(microsecond=0).isoformat()


def _meta_get(key: str, db_path=None) -> str | None:
    c = _connect(db_path)
    try:
        row = c.execute("SELECT value FROM meta WHERE key = ?",
                        (key,)).fetchone()
    finally:
        c.close()
    return None if row is None else row[0]


def _meta_set(key: str, value: str | None, db_path=None) -> None:
    c = _connect(db_path)
    try:
        if value is None:
            c.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                      (key, value))
        c.commit()
    finally:
        c.close()


def latch_state(db_path=None) -> dict[str, Any] | None:
    """The tripped latch record, or None when the latch is CLEAR.

    Fails CLOSED: a latch that cannot be read reports itself as tripped
    (reason LATCH_UNREADABLE), because "I could not check whether mutation
    is still believed safe" must never read as "it is"."""
    try:
        raw = _meta_get(LATCH_KEY, db_path)
    except (sqlite3.Error, OSError) as exc:      # fail closed
        logger.warning("queue hygiene latch unreadable: %s", exc)
        return {"reason": LATCH_UNREADABLE, "error": str(exc)[:200]}
    if raw is None:
        return None
    try:
        rec = json.loads(raw)
    except (TypeError, ValueError):
        rec = None
    if not isinstance(rec, dict):
        # A corrupt latch row is still a latch row: someone tripped it.
        return {"reason": LATCH_UNREADABLE, "raw": str(raw)[:200]}
    return rec


def trip_latch(reason: str, *, queue_id: str, pass_id: str,
               detail: dict[str, Any] | None = None,
               now: datetime | None = None, db_path=None) -> dict[str, Any]:
    """Disable all future mutation until an operator clears the latch.

    The record keeps exactly what an operator needs to decide whether to
    re-arm: when, why, which queue and pass, and -- for a protected-sequence
    violation -- WHICH ids went missing or moved, capped, plus what this
    pass itself removed, so the known false positive (a listener deleting a
    song mid-pass) can be told apart from damage. No titles, no URIs, no
    queue snapshot."""
    rec = {"tripped_at": _iso(now), "reason": reason, "queue_id": queue_id,
           "pass_id": pass_id, "detail": dict(detail or {})}
    try:
        _meta_set(LATCH_KEY, json.dumps(rec, sort_keys=True), db_path)
    except (sqlite3.Error, OSError):
        # The pass has already stopped; the danger is a FUTURE pass. Log
        # loudly. latch_state fails closed on an unreadable row, but a row
        # that was never written is simply absent -- so this is the one
        # gap, and it is reported as such.
        logger.error("queue hygiene: latch trip could not be persisted "
                     "(reason=%s queue=%s)", reason, queue_id, exc_info=True)
    logger.error("queue_hygiene_latch_tripped reason=%s queue=%s pass=%s %s",
                 reason, queue_id, pass_id[:8], json.dumps(detail or {},
                                                            sort_keys=True))
    return rec


def clear_latch(db_path=None) -> bool:
    """Re-arm: an EXPLICIT operator (or test) action. Never called by the
    engine, never on restart. Returns whether a latch was cleared."""
    was = latch_state(db_path) is not None
    _meta_set(LATCH_KEY, None, db_path)
    if was:
        logger.warning("queue_hygiene_latch_cleared")
    return was


def _last_pass_at(db_path=None) -> datetime | None:
    try:
        raw = _meta_get(LAST_PASS_KEY, db_path)
        return datetime.fromisoformat(raw) if raw else None
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return None


def _note_pass(now: datetime, db_path=None) -> None:
    _meta_set(LAST_PASS_KEY, _iso(now), db_path)


# --- lookups (Parlorwave's own durable state; injectable) ------------------------

def classification_context(deps: dict[str, Any] | None = None,
                           db_path=None) -> dict[str, Callable]:
    """THE PUBLIC, READ-ONLY HALF OF THIS ENGINE.

    Ownership and lifecycle are one question with one answer, and two
    consumers need it: this module, deciding whether it may remove an item,
    and presentation, deciding what to call one. The second must not restate
    the first -- a drawer that decided "spent" by its own rules would be a
    second classifier, free to drift from the one that deletes.

    So the pair `classification_context` + `classify_item` is published, and
    it is ALL that is published of the engine. Building a context reads
    nothing: it resolves the six questions classification may ask into
    callables (lazily, so this module still imports neither the DJ engine
    nor the newsroom at load time), and every one of them is a read of
    Parlorwave's own durable state. Nothing here can queue, delete,
    terminalize, retire, latch, or reach Music Assistant at all, and a
    caller holding a context has no route to any of those.

    Build ONE context for a batch of items and reuse it: the I/O is in the
    lookups, which classify_item calls only for an item whose URI carries a
    program route, and never for music.

    `deps` takes the same `*_fn` seams the engine's own callers inject.
    """
    return _lookups(deps or {}, db_path)


def _lookups(deps: dict[str, Any], db_path) -> dict[str, Callable]:
    """The six questions classification may ask, resolved lazily so this
    module does not import the DJ engine or the newsroom at load time.

    The engine's own half calls this directly; everything outside the
    engine goes through `classification_context`, which is the same thing
    under the name that says it is public and that it reads."""
    def dj_row(token):
        import dj_engine
        return dj_engine.get_break(token, db_path)

    def news_row(token):
        import news_bulletins
        return news_bulletins.get_bulletin(token, db_path)

    def dj_gate(token):
        import dj_engine
        return dj_engine.audio_available(token, db_path)

    def news_gate(token):
        import news_bulletins
        return news_bulletins.audio_available(token, db_path)

    def locked(token):
        import news_schedule
        return bool(news_schedule.bulletin_locked(token)[0])

    return {
        "dj_row": deps.get("dj_row_fn") or dj_row,
        "news_row": deps.get("news_row_fn") or news_row,
        "retired_kind": deps.get("retired_kind_fn")
        or (lambda t: program_tokens.retired_kind(t, db_path)),
        "dj_gate": deps.get("dj_gate_fn") or dj_gate,
        "news_gate": deps.get("news_gate_fn") or news_gate,
        "locked": deps.get("locked_fn") or locked,
    }


GATE_LIVE = "live"
GATE_RETIRED = "retired"
GATE_UNKNOWN = "unknown"


def gate_verdict(result: Any) -> str:
    """Translate an audio_available() answer into the three words that
    matter here. 200 + the tombstone asset, or 410 (known token, asset
    missing), is RETIRED; 200 with anything else is LIVE audio; everything
    else (404, garbage) is UNKNOWN. Live and unknown both stop a delete."""
    try:
        status, path = result
    except (TypeError, ValueError):
        return GATE_UNKNOWN
    if status == 200:
        return GATE_RETIRED if program_audio.is_tombstone(path) else GATE_LIVE
    if status == 410:
        return GATE_RETIRED
    return GATE_UNKNOWN


# --- classification (ownership + lifecycle) --------------------------------------

def item_uri(item: dict[str, Any]) -> str:
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
    return str(media.get("uri") or "")


#: A malformed item (a nested field that is truthy but not a dict) is the
#: only way the label can fail, and an unlabelled item is still a fully
#: protected item -- so this is caught NARROWLY, by exception type. The
#: module's "no broad except" rule is a real invariant, pinned by test:
#: an operational failure must be _OPERATIONAL and a defect must propagate.
_LABEL_ERRORS = (AttributeError, TypeError, ValueError)


def _provenance(item: dict[str, Any]) -> str:
    """Which family a NON-program item belongs to, as a bare word.

    Reporting only. This label never decides anything: an item with no
    program route is protected because it has no program route, whatever
    word lands here."""
    try:
        import source_capabilities
        return str(source_capabilities.item_provenance(item) or "unknown")
    except _LABEL_ERRORS:
        return "unknown"


def _news_beyond_age(row: dict[str, Any]) -> bool | None:
    """True/False when the row's date is readable, None when it is not.
    The number is the planner's, read through news_bulletins so the three
    users of it cannot drift."""
    import news_bulletins
    published = row.get("published_at")
    if news_bulletins._age_minutes(published) is None:
        return None
    return bool(news_bulletins._beyond_bulletin_age(published))


def classify_item(item: dict[str, Any], *, queue_id: str,
                  lookups: dict[str, Callable]) -> dict[str, Any]:
    """Ownership and lifecycle for ONE queue item. No position here -- see
    classify_future for the buffer zone.

    Returns {queue_item_id, kind, token, verdict, reason, ownership,
    lifecycle}. Nothing about the item but its id and (for a route-shaped
    URI) its kind and token is kept."""
    qid = str(item.get("queue_item_id") or "")
    out = {"queue_item_id": qid, "kind": "", "token": "",
           "verdict": V_PRESERVE, "reason": "music_or_provider",
           "ownership": "", "lifecycle": "", "provenance": ""}
    kind_raw, token = media_ref.programming_ref(item_uri(item))
    if not kind_raw:
        # Not ours: the customer's music, a manual add, a request, or the
        # provider's own pick. It can never become a candidate, and the
        # only thing kept about it is which of those families it belongs
        # to -- so a recovery can SAY what it preserved. Music Assistant
        # exposes no "this was asked for by hand" marker, so a manual add
        # and a refill both read as local_owned_music; both are protected
        # by the same rule, which is the rule that matters.
        out["provenance"] = _provenance(item)
        return out
    kind = KIND_DJ if kind_raw == "break" else KIND_NEWS
    out["kind"] = kind
    if not _TOKEN_RE.match(token or ""):
        out.update(verdict=V_UNKNOWN, reason="malformed_token", token="")
        return out
    out["token"] = token
    if kind == KIND_DJ:
        return _classify_dj(out, token, queue_id, lookups)
    return _classify_news(out, token, lookups)


def _classify_dj(out, token, queue_id, lookups):
    row = lookups["dj_row"](token)
    if row is not None:
        state = str(row.get("state") or "")
        if str(row.get("queue_id") or "") != queue_id:
            # A real break of ours -- on some other queue. Not this pass's
            # to judge; ambiguity is preserved, not resolved.
            out.update(verdict=V_PRESERVE, reason="dj_queue_mismatch",
                       ownership=OWN_DJ_ROW, lifecycle=state)
            return out
        if state in DJ_TERMINAL_STATES:
            out.update(verdict=V_DELETE, reason=f"dj_terminal:{state}",
                       ownership=OWN_DJ_ROW, lifecycle=state)
            return out
        if state in DJ_LIVE_STATES:
            out.update(verdict=V_PRESERVE, reason=f"dj_live:{state}",
                       ownership=OWN_DJ_ROW, lifecycle=state)
            return out
        out.update(verdict=V_PRESERVE, reason="dj_unknown_state",
                   ownership=OWN_DJ_ROW, lifecycle=state)
        return out
    retired = lookups["retired_kind"](token)
    if retired == KIND_DJ:
        out.update(verdict=V_DELETE, reason="dj_retired_ledger",
                   ownership=OWN_DJ_LEDGER, lifecycle="retired")
    elif retired is not None:
        out.update(verdict=V_UNKNOWN, reason="ledger_kind_mismatch",
                   lifecycle="retired")
    else:
        out.update(verdict=V_UNKNOWN, reason="unknown_token")
    return out


def _classify_news(out, token, lookups):
    # THE ABSOLUTE VETO comes first: a bulletin the schedule has committed
    # to air (queued, intro airing, bulletin airing) is untouchable whatever
    # its row says, because its URL is about to be requested.
    if lookups["locked"](token):
        out.update(verdict=V_PRESERVE, reason="bulletin_locked",
                   ownership=OWN_NEWS_ROW, lifecycle="locked")
        return out
    row = lookups["news_row"](token)
    retired = lookups["retired_kind"](token)
    if row is not None:
        state = str(row.get("state") or "")
        if state in NEWS_STALE_STATES:
            out.update(verdict=V_DELETE, reason=f"news_{state}",
                       ownership=OWN_NEWS_ROW, lifecycle=state)
            return out
        if state == "cached":
            if retired == KIND_NEWS:
                # Spent beats cached (news_bulletins.audio_available): the
                # row is retention history, the ledger says it aired.
                out.update(verdict=V_DELETE, reason="news_cached_retired",
                           ownership=OWN_NEWS_LEDGER, lifecycle="cached")
                return out
            beyond = _news_beyond_age(row)
            if beyond is True and \
                    gate_verdict(lookups["news_gate"](token)) == GATE_RETIRED:
                out.update(verdict=V_DELETE, reason="news_cached_stale",
                           ownership=OWN_NEWS_ROW, lifecycle="cached")
                return out
            out.update(verdict=V_PRESERVE,
                       reason=("news_cached_fresh" if beyond is False
                               else "news_cached_undated"),
                       ownership=OWN_NEWS_ROW, lifecycle="cached")
            return out
        if state in NEWS_LIVE_STATES:
            out.update(verdict=V_PRESERVE, reason=f"news_live:{state}",
                       ownership=OWN_NEWS_ROW, lifecycle=state)
            return out
        out.update(verdict=V_PRESERVE, reason="news_unknown_state",
                   ownership=OWN_NEWS_ROW, lifecycle=state)
        return out
    if retired == KIND_NEWS:
        out.update(verdict=V_DELETE, reason="news_retired_ledger",
                   ownership=OWN_NEWS_LEDGER, lifecycle="retired")
    elif retired is not None:
        out.update(verdict=V_UNKNOWN, reason="ledger_kind_mismatch",
                   lifecycle="retired")
    else:
        out.update(verdict=V_UNKNOWN, reason="unknown_token")
    return out


def _zone(position: int, index_in_buffer: Any) -> str:
    if not isinstance(index_in_buffer, int) or isinstance(index_in_buffer, bool):
        return ZONE_UNKNOWN
    return ZONE_FUTURE if position > index_in_buffer else ZONE_BUFFERED


def _action(verdict: str, zone: str) -> str:
    if verdict == V_DELETE:
        return ACT_WOULD_DELETE if zone == ZONE_FUTURE else ACT_PROTECTED
    return ACT_LEAVE_UNKNOWN if verdict == V_UNKNOWN else ACT_PRESERVE


def classify_future(future: list[dict[str, Any]], *, queue_id: str,
                    current_position: int, index_in_buffer: Any,
                    lookups: dict[str, Callable]) -> list[dict[str, Any]]:
    """Every future item, in queue order, with its absolute list position
    (current_position + 1 + offset), its buffer zone and its action."""
    out = []
    for offset, item in enumerate(future):
        rec = classify_item(item, queue_id=queue_id, lookups=lookups)
        pos = current_position + 1 + offset
        zone = _zone(pos, index_in_buffer)
        rec.update(position=pos, offset=offset, zone=zone,
                   action=_action(rec["verdict"], zone))
        out.append(rec)
    return out


def summarize(classified: list[dict[str, Any]]) -> dict[str, int]:
    counts = {ACT_WOULD_DELETE: 0, ACT_PRESERVE: 0, ACT_LEAVE_UNKNOWN: 0,
              ACT_PROTECTED: 0, "music_or_provider": 0, "program": 0}
    for rec in classified:
        counts[rec["action"]] += 1
        if rec["kind"]:
            counts["program"] += 1
        else:
            counts["music_or_provider"] += 1
    return counts


def preserved_families(classified: list[dict[str, Any]]) -> dict[str, int]:
    """How many NON-program future items each family contributed, so a
    recovery report can name what it protected without a queue dump."""
    out: dict[str, int] = {}
    for rec in classified:
        if rec["kind"]:
            continue
        fam = rec.get("provenance") or "unknown"
        out[fam] = out.get(fam, 0) + 1
    return out


def retained_program_reasons(classified: list[dict[str, Any]]) -> dict[str, int]:
    """Reasons why program items were retained rather than deleted."""
    out: dict[str, int] = {}
    for rec in classified:
        if not rec.get("kind"):
            continue
        if rec.get("action") == ACT_WOULD_DELETE:
            continue
        reason = str(rec.get("reason") or "unknown")
        out[reason] = out.get(reason, 0) + 1
    return out



# --- the protected-sequence invariant ---------------------------------------------

def check_subsequence(protected: list[str], observed: list[str]
                      ) -> dict[str, Any]:
    """Is `protected` a subsequence of `observed`, in identical relative
    order? Appends, insertions and removals of OTHER ids are all fine.

    Returns {ok, missing, reordered}: `missing` are protected ids absent
    from `observed` entirely; `reordered` are present ids that appear
    earlier than an id that preceded them in `protected` (each descent is
    named once)."""
    where = {qid: i for i, qid in enumerate(observed)}
    missing = [p for p in protected if p not in where]
    reordered: list[str] = []
    high = -1
    for p in protected:
        if p not in where:
            continue
        idx = where[p]
        if idx < high:
            reordered.append(p)
        else:
            high = idx
    return {"ok": not missing and not reordered, "missing": missing,
            "reordered": reordered}


# --- queue reads (ma_queue is the only transport) -------------------------------
#
# FOUR SHAPES OF READ, each as small as the fact it proves:
#
#   _read              the complete FUTURE: the snapshot, and the fallback
#                      when a candidate has drifted out of its window
#   _probe             queue_state + a ONE-item page at current_index: head
#                      identity, both indexes, state, and proof that
#                      current_index is still the head's list position
#   _locate            a (2 * LOCATE_WINDOW + 1)-item page around a
#                      candidate's last known position
#   _read_whole_queue  every id in the queue, history included: the closing
#                      invariant, and the only read that can prove a
#                      deleted id is gone from EVERYWHERE
#
# `offset` is a LIST POSITION and `limit` is honoured exactly (see
# ma_queue), which is what makes the two small shapes sound: an item found at
# page index i of a page requested at offset o sits at list position o + i.
# Every read counts itself in res["reads"] so a test can pin the cost.

_OPERATIONAL = (ma_queue.MAQueueError, sqlite3.Error, OSError)


def _items_fn(deps: dict[str, Any]) -> Callable:
    return deps.get("items_fn") or (
        lambda q, limit, offset: ma_queue.list_queue_items(
            q, limit=limit, offset=offset))


def _count(res: dict[str, Any] | None, kind: str, n: int = 1) -> None:
    if res is not None:
        res["reads"][kind] = res["reads"].get(kind, 0) + n


def _state(queue_id: str, deps: dict[str, Any]) -> dict[str, Any]:
    state_fn = deps.get("state_fn") or ma_queue.queue_state
    state = state_fn(queue_id) or {}
    head = (state.get("current_item") or {}).get("queue_item_id")
    # The item Music Assistant names as NEXT. MA can commit it to the player
    # while it is still beyond index_in_buffer, so no position check
    # protects it (see the last look in _mutate). None when MA names nothing.
    nxt = state.get("next_item")
    return {"head": head, "current_index": state.get("current_index"),
            "index_in_buffer": state.get("index_in_buffer"),
            "state": state.get("state"), "total": state.get("items"),
            "next": nxt.get("queue_item_id") if isinstance(nxt, dict) else None,
            # MA's Autoplay switch, read only (None when not reported): what
            # appended a spent tail, for autoplay_watch
            "autoplay": state.get("autoplay_enabled")}


def _read(queue_id: str, deps: dict[str, Any], res: dict[str, Any] | None = None
          ) -> tuple[dict | None, str]:
    """One COMPLETE read of the future: (evidence, '') or (None, reason).
    Evidence: {head, current_index, index_in_buffer, state, total,
    current_position, anchored, future, ids}."""
    future_fn = deps.get("future_fn") or ma_queue.read_future_items
    try:
        ev = _state(queue_id, deps)
    except ma_queue.MAQueueError as exc:
        return None, f"state_unreadable:{str(exc)[:120]}"
    finally:
        _count(res, "state")
    if not ev["head"]:
        return None, "no_head_identity"
    try:
        read = future_fn(queue_id, ev["head"],
                         current_index=ev["current_index"],
                         total_items=ev["total"])
    except ma_queue.MAQueueError as exc:
        return None, f"future_unreadable:{str(exc)[:120]}"
    finally:
        _count(res, "full")
    if not read.get("complete"):
        return None, f"future_incomplete:{read.get('incomplete_reason') or ''}"
    future = list(read.get("future") or [])
    pos = read.get("current_position")
    ev.update(current_position=pos, future=future,
              ids=[str(it.get("queue_item_id") or "") for it in future],
              anchored=_same_int(ev["current_index"], pos))
    return ev, ""


def _probe(queue_id: str, deps: dict[str, Any], res: dict[str, Any] | None
           ) -> tuple[dict | None, str]:
    """The cheap revalidation: head identity, both indexes and state from
    queue_state, plus a one-item page at current_index that must return the
    head -- the same anchoring proof the complete read makes, for one item
    instead of the whole future."""
    try:
        ev = _state(queue_id, deps)
        _count(res, "state")
        if not ev["head"]:
            return None, "no_head_identity"
        cur = ev["current_index"]
        anchored = False
        if isinstance(cur, int) and not isinstance(cur, bool) and cur >= 0:
            page = _items_fn(deps)(queue_id, 1, cur)
            _count(res, "pages")
            anchored = bool(page) and page[0].get("queue_item_id") == ev["head"]
    except ma_queue.MAQueueError as exc:
        return None, f"probe_failed:{str(exc)[:120]}"
    ev["anchored"] = anchored
    return ev, ""


def _locate(queue_id: str, queue_item_id: str, hint: int,
            deps: dict[str, Any], res: dict[str, Any] | None
            ) -> tuple[int | None, dict[str, Any] | None]:
    """(list position, item) of one item near its last known position, or
    (None, None) when it is not within LOCATE_WINDOW of the hint. Raises
    MAQueueError; the caller decides what an unreadable page means."""
    start = max(0, int(hint) - LOCATE_WINDOW)
    page = _items_fn(deps)(queue_id, 2 * LOCATE_WINDOW + 1, start)
    _count(res, "pages")
    for i, it in enumerate(page):
        if it.get("queue_item_id") == queue_item_id:
            return start + i, it
    return None, None


def _read_whole_queue(queue_id: str, deps: dict[str, Any], *,
                      res: dict[str, Any] | None = None,
                      page: int = 500) -> list[str] | None:
    """Every queue_item_id in the queue, history included, in order -- the
    closing-check read. Explicit limit on every page (MA truncates a
    limit-less read at 500 in silence). None when it cannot be completed."""
    items_fn = _items_fn(deps)
    ids: list[str] = []
    offset = 0
    _count(res, "full")
    try:
        while offset < WHOLE_QUEUE_CAP:
            chunk = items_fn(queue_id, page, offset)
            if not chunk:
                return ids
            ids.extend(str(it.get("queue_item_id") or "") for it in chunk)
            offset += len(chunk)
            if len(chunk) < page:
                return ids
    except ma_queue.MAQueueError as exc:
        logger.warning("queue hygiene: closing read failed on %s: %s",
                       queue_id, exc)
        return None
    return None


def _same_int(a: Any, b: Any) -> bool:
    return (isinstance(a, int) and not isinstance(a, bool)
            and isinstance(b, int) and not isinstance(b, bool) and a == b)


def _units_consistent(ev: dict[str, Any]) -> bool:
    """index_in_buffer is compared with LIST positions, which is only sound
    while MA's current_index IS the head's list position -- unproven in
    general (ma_queue.read_future_items), so proven per read here."""
    return bool(ev.get("anchored"))


# --- authority and exclusions -----------------------------------------------------

def _default_authority(queue_id: str) -> str:
    """Why this queue is Parlorwave's to act on: the active station's queue
    ('station'), or a queue with an active playback attachment ('session').
    Empty = not ours, and so is an unreadable store."""
    try:
        import station_store
        st = station_store.get_active_station()
        if st and str(st.get("queue_id") or "") == queue_id:
            return "station"
        import playback
        if playback.get_active_for_queue(queue_id):
            return "session"
    except (sqlite3.Error, OSError):
        pass
    return ""


def _prior_attachment(queue_id: str,
                      db_path: Path | None = None) -> tuple[bool, dict[str, Any] | None]:
    """Check whether this queue has proven prior Parlorwave ownership/attachment:
    either a prior session in sessions.db, a prior/active station in station_store,
    or a record in queue_ownership."""
    if not queue_id:
        return False, None
    try:
        import playback
        playback.ensure_sessions_schema(db_path)
        c = playback._connect(db_path)
        try:
            row = c.execute(
                f"SELECT {', '.join(playback._SESSION_COLS)} FROM sessions "
                "WHERE queue_id = ? ORDER BY started_at DESC LIMIT 1",
                (queue_id,)
            ).fetchone()
            if row:
                return True, playback._session_dict(row)
        finally:
            c.close()
    except (sqlite3.Error, OSError):
        pass
    try:
        import station_store
        st = station_store.get_active_station(db_path=db_path)
        if st and str(st.get("queue_id") or "") == queue_id:
            return True, {"queue_id": queue_id,
                          "player": (st.get("program") or {}).get("target", {}).get("entity_id")}
    except (sqlite3.Error, OSError):
        pass
    try:
        import queue_ownership
        rec = queue_ownership.get_record(queue_id, db_path=db_path)
        if rec and rec.get("state") in (queue_ownership.OWNED, queue_ownership.UNVERIFIED,
                                        queue_ownership.CONTESTED, queue_ownership.RELEASED):
            return True, {"queue_id": queue_id, "player": ""}
    except (sqlite3.Error, OSError):
        pass
    return False, None


def _check_recovery_authority(queue_id: str,
                             recovery_evidence: dict[str, Any] | None = None,
                             db_path: Path | None = None) -> tuple[bool, str]:
    """Recovery-only authority gate: allows a post-quiet recovery pass (quiet_recovery
    or position_regressed after a long quiet stretch) to proceed without an active
    playback session, but ONLY when independent authoritative evidence proves this queue
    and player belong to Parlorwave and are actually playing."""
    if not queue_id:
        return False, "no_queue_id"
    if not recovery_evidence or not isinstance(recovery_evidence, dict):
        return False, "no_recovery_evidence"

    # 1. Trigger must be an explicit recovery trigger
    trigger = str(recovery_evidence.get("trigger") or "")
    if trigger not in ("quiet_recovery", "position_regressed"):
        return False, "not_recovery_trigger"

    # If trigger is position_regressed, it must have quiet-recovery evidence
    if trigger == "position_regressed" and not (
            recovery_evidence.get("after_quiet")
            or recovery_evidence.get("quiet_recovery")
            or (recovery_evidence.get("quiet_s") is not None and recovery_evidence.get("quiet_s") >= 120.0)):
        return False, "not_after_quiet"

    # 2. Target queue must match the queue being observed
    ev_queue = str(recovery_evidence.get("queue_id") or "").strip()
    if not ev_queue or ev_queue != queue_id:
        return False, "queue_mismatch"

    # 3. Player/queue must actually be playing
    if recovery_evidence.get("state") != "playing":
        return False, "not_playing"

    # 4. Observer worker must be authoritative for playback feeding
    if not recovery_evidence.get("feeding"):
        return False, "observer_not_feeding"

    # 5. Prior Parlorwave ownership/attachment proof
    has_prior, prior_rec = _prior_attachment(queue_id, db_path=db_path)
    if not has_prior:
        return False, "no_prior_ownership"

    # 6. Player identity match if specified
    ev_player = str(recovery_evidence.get("player_id") or "").strip()
    prior_player = str((prior_rec or {}).get("player") or "").strip()
    if ev_player and prior_player:
        if ev_player != prior_player and ev_player != queue_id and prior_player != queue_id:
            return False, "player_mismatch"

    return True, "recovery_authoritative"


def _default_exclusion(queue_id: str) -> str:
    """A reason mutation must wait, or ''. A pending refill batch means the
    refill path's pre-tail proof is live and a deletion near the tail could
    invalidate it (a duplicate or refused append is worse than a stale
    item); a preparing station means an external queue rewrite may be in
    flight. Unreadable state is an exclusion, not a permission."""
    try:
        import station_store
        if station_store.get_preparing_station() is not None:
            return "start_radio_in_flight"
        st = station_store.get_active_station()
        if st and st.get("station_id") and \
                station_store.get_pending_batch(st["station_id"]) is not None:
            return "refill_pending"
    except (sqlite3.Error, OSError) as exc:
        return f"station_state_unreadable:{str(exc)[:80]}"
    return ""


@contextmanager
def _default_lock(timeout: float):
    with station_lock.station_mutation(timeout=timeout):
        yield


# --- the pass -----------------------------------------------------------------------

def _result(queue_id: str, pass_id: str, mode: str) -> dict[str, Any]:
    return {"pass_id": pass_id, "queue_id": queue_id, "mode": mode,
            "policy": POLICY_ROUTINE, "permission_bypassed": False,
            "gap_waived": False,
            "outcome": "", "reason": "", "authority": "",
            "ordinary_authority": "", "recovery_only_authority": False,
            "recovery_authoritative": False, "recovery_reason": "",
            "recovery_evidence": None,
            "preconditions": {}, "snapshot": None,
            "candidates": 0, "protected_count": 0, "watched_count": 0,
            "preserved": {}, "retained_reasons": {},
            "deleted": [], "unconfirmed": [], "skipped": [], "attempted": 0,
            "remaining": 0, "more_remaining": False,
            "watched_anomalies": None,
            "latch_tripped": False, "counts": {}, "classified": [],
            "reads": {"state": 0, "pages": 0, "full": 0},
            "lock_held_s": 0.0, "elapsed_s": 0.0}


def dry_run(queue_id: str, *, db_path=None, deps: dict[str, Any] | None = None
            ) -> dict[str, Any]:
    """Classify the live future of a queue and report what a mutating pass
    WOULD do. Performs no delete, takes no lock, writes nothing. The
    preconditions are evaluated and reported, never enforced."""
    return run_pass(queue_id, dry_run=True, db_path=db_path, deps=deps)


def run_pass(queue_id: str, *, dry_run: bool = False,
             mandatory: bool = False, ignore_gap: bool = False,
             now: datetime | None = None, db_path=None,
             deps: dict[str, Any] | None = None,
             limits: Limits | None = None,
             recovery_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """One hygiene pass over `queue_id`. Returns a result dict.

    `mandatory` selects POLICY_MANDATORY: the `playback.queue_hygiene`
    permission is not asked, because the caller has positive evidence of a
    queue-position regression -- a lifecycle fault in which already-aired
    program items are in the future again. NOTHING ELSE CHANGES. The
    latch, authority, the station lock, the refill/Start Radio exclusions,
    the minimum gap, the playing state, the buffer, positive ownership,
    positive staleness, the serving belt and the protected-sequence check
    all apply exactly as they do to a routine pass, and the only removal
    verb is still one delete of one positively-owned, positively-stale
    item by exact id. A mandatory pass can therefore still come back
    latched, refused, lock_busy, excluded or min_gap.

    `ignore_gap` waives MIN_PASS_GAP_S FOR THIS PASS ONLY, and nothing
    else. It exists for one caller: the bounded stabilization chain in
    queue_hygiene_triggers, which must be able to finish a reconciliation
    that an earlier pass started and could not complete. A rewind can
    arrive as two regressions seconds apart -- the first aborts on
    `head_changed` with the head still moving, having already stamped the
    gap on its first delete, and the second, which sees the settled queue
    and the whole candidate set, is refused `min_gap`. The belt does its
    job (it is a storm belt) and the cleanup still does not converge.
    Waiving it for a chained continuation keeps the belt for
    every INDEPENDENT event while letting one event finish its own work.

    It is not a general escape hatch: the caller must already hold a
    reconciliation chain, the pass is still bounded by
    MAX_DELETES_PER_PASS and PASS_TIME_BUDGET_S, and every other gate --
    permission (unless mandatory), latch, authority, the station mutation
    lock, the refill / Start Radio exclusions, the playing state, the
    buffer, positive ownership, positive staleness, the serving belt and
    the protected-sequence check -- is asked exactly as always.

    `recovery_evidence` is THIS CALLER'S proof of a post-quiet recovery
    event (the observer's, through the trigger module's request, or the
    one an earlier pass of the same chain validated, through
    maybe_continue). It is judged by _check_recovery_authority and, when
    it holds, reported back in the result so the episode can keep it. A
    caller that brings none gets none: there is no process-wide source.

    Operational failures (Music Assistant, the sessions database, lock
    contention) come back as fail-closed results; a programming defect
    propagates. Called by queue_hygiene_triggers (one pass per
    qualifying observer event), by the manual route, by the dry-run
    diagnostic and by tests."""
    deps = dict(deps or {})
    lim = limits or Limits()
    clock = deps.get("clock") or time.monotonic
    now_dt = now or datetime.now()
    pass_id = uuid.uuid4().hex
    res = _result(queue_id, pass_id, "dry_run" if dry_run else "mutate")
    res["policy"] = POLICY_MANDATORY if mandatory else POLICY_ROUTINE
    res["gap_waived"] = bool(ignore_gap)
    rec_ev = recovery_evidence if recovery_evidence is not None else deps.get("recovery_evidence")
    t0 = clock()
    try:
        _run(queue_id, res, dry_run=dry_run, mandatory=mandatory,
             ignore_gap=ignore_gap, now_dt=now_dt, db_path=db_path,
             deps=deps, lim=lim, clock=clock, t0=t0,
             recovery_evidence=rec_ev)
    except _OPERATIONAL as exc:
        logger.warning("queue hygiene pass %s stopped: %s", pass_id[:8], exc,
                       exc_info=True)
        res["outcome"], res["reason"] = "error", str(exc)[:200]
    res["elapsed_s"] = round(clock() - t0, 3)
    if not dry_run:
        logger.info("queue_hygiene_pass queue=%s policy=%s outcome=%s "
                    "reason=%s deleted=%d skipped=%d candidates=%d "
                    "remaining=%d lock_held_s=%.3f reads=%s%s%s",
                    queue_id, res["policy"], res["outcome"], res["reason"],
                    len(res["deleted"]), len(res["skipped"]),
                    res["candidates"], res["remaining"], res["lock_held_s"],
                    res["reads"],
                    " (permission bypassed: confirmed queue regression)"
                    if res["permission_bypassed"] else "",
                    " (recovery-only authority)"
                    if res.get("recovery_only_authority") else "")
    return res


def _preconditions(queue_id: str, *, now_dt: datetime, db_path, deps,
                   lim: Limits,
                   recovery_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    settings_fn = deps.get("settings_fn")
    if settings_fn is None:
        import settings
        settings_fn = settings.queue_hygiene_permitted   # never raises
    permitted = bool(settings_fn())
    latch = latch_state(db_path)

    authority_fn = deps.get("authority_fn")
    if authority_fn is not None:
        ordinary_authority = authority_fn(queue_id)
    else:
        ordinary_authority = _default_authority(queue_id)

    # Recovery evidence is CALL-LOCAL: the caller's own argument, or the
    # dependency injected for this call. The engine asks no module for it.
    # A fallback to a process-wide dictionary the observer filled around
    # its request would let any other pass on the same queue in that
    # window -- a dry-run diagnostic, the manual route, a timer -- inherit
    # the event's proof and, with it, recovery-only authority it had not
    # earned. Authority belongs to the request that brought the evidence.
    recovery_evidence = deps.get("recovery_evidence") or recovery_evidence
    recovery_ok, recovery_reason = False, ""
    recovery_fn = deps.get("recovery_authority_fn")
    if recovery_fn is not None:
        recovery_ok = bool(recovery_fn(queue_id))
        recovery_reason = "recovery_authority_injected" if recovery_ok else "recovery_authority_refused"
    elif recovery_evidence is not None:
        recovery_ok, recovery_reason = _check_recovery_authority(
            queue_id, recovery_evidence=recovery_evidence, db_path=db_path)

    authority = ordinary_authority
    recovery_only = False
    if not authority and recovery_ok:
        authority = "recovery"
        recovery_only = True

    exclusion = (deps.get("exclusion_fn") or _default_exclusion)(queue_id)
    last = _last_pass_at(db_path)
    gap_ok = True
    if last is not None:
        # Never a negative gap across a DST
        # fall-back, never held past the true elapsed time.
        bounds = service_clock.age_bounds(last, now_dt)
        gap_ok = bounds is not None and bounds[1] >= lim.min_gap_s
    return {"permitted": permitted, "latch": latch, "authority": authority,
            "ordinary_authority": ordinary_authority,
            "recovery_only_authority": recovery_only,
            "recovery_authoritative": recovery_ok,
            "recovery_reason": recovery_reason,
            # The evidence this pass VALIDATED, for the caller to keep with
            # the episode: a chain armed after a recovery pass carries
            # it, so each continuation is authorized -- and re-checked here
            # -- on the same proof. Never an injected verdict, never stale.
            "recovery_evidence": (dict(recovery_evidence)
                                  if recovery_ok and isinstance(
                                      recovery_evidence, dict) else None),
            "exclusion": exclusion, "last_pass_at": None if last is None
            else _iso(last), "gap_ok": gap_ok}


def _run(queue_id, res, *, dry_run, now_dt, db_path, deps, lim, clock, t0,
         mandatory=False, ignore_gap=False, recovery_evidence=None):
    if not queue_id:
        res["outcome"], res["reason"] = "refused", "no_queue_id"
        return
    pre = _preconditions(queue_id, now_dt=now_dt, db_path=db_path, deps=deps,
                         lim=lim, recovery_evidence=recovery_evidence)
    res["recovery_evidence"] = pre.pop("recovery_evidence", None)
    res["preconditions"] = {**pre, "latch": pre["latch"] is not None}
    res["authority"] = pre["authority"]
    res["ordinary_authority"] = pre.get("ordinary_authority", "")
    res["recovery_only_authority"] = bool(pre.get("recovery_only_authority", False))
    res["recovery_authoritative"] = bool(pre.get("recovery_authoritative", False))
    res["recovery_reason"] = pre.get("recovery_reason", "")

    if dry_run:
        _dry(queue_id, res, deps=deps, db_path=db_path)
        return

    # --- fail-closed gates, in the order they are cheapest to answer -------
    # The PERMISSION gate, and the one case that does not ask it. Routine
    # cleanup is the operator's call. A confirmed regression is not
    # cleanup: spent programming is ahead of the head and will re-air, so
    # ownership reconciliation is owed whatever the preference says. The
    # bypass is recorded, never silent, and nothing below it is relaxed.
    if not pre["permitted"]:
        if not mandatory:
            res["outcome"], res["reason"] = "disabled", "queue_hygiene_off"
            return
        res["permission_bypassed"] = True
    if pre["latch"] is not None:
        res["outcome"] = "latched"
        res["reason"] = str(pre["latch"].get("reason") or "tripped")
        return
    if not pre["authority"]:
        res["outcome"], res["reason"] = "refused", "not_authoritative"
        return
    lock = deps.get("lock_fn") or _default_lock
    try:
        held = lock(lim.lock_wait_s)
        held.__enter__()
    except station_lock.StationMutationBusy as exc:
        res["outcome"], res["reason"] = "lock_busy", str(exc)[:120]
        return
    t_lock = clock()
    try:
        _mutate(queue_id, res, now_dt=now_dt, db_path=db_path, deps=deps,
                lim=lim, clock=clock, t0=t0, ignore_gap=ignore_gap)
    finally:
        res["lock_held_s"] = round(clock() - t_lock, 3)
        held.__exit__(None, None, None)


def _dry(queue_id, res, *, deps, db_path):
    ev, why = _read(queue_id, deps, res)
    if ev is None:
        res["outcome"], res["reason"] = "unreadable", why
        return
    res["snapshot"] = _snapshot_brief(ev)
    lookups = _lookups(deps, db_path)
    classified = classify_future(
        ev["future"], queue_id=queue_id,
        current_position=ev["current_position"],
        index_in_buffer=ev["index_in_buffer"], lookups=lookups)
    # The belt, reported per would-delete candidate so a dry run can show a
    # lifecycle/serving disagreement before any pass could meet it; and the
    # candidate Music Assistant names as next, which a pass holds back.
    for rec in classified:
        if rec["action"] == ACT_WOULD_DELETE:
            rec["gate"] = _belt(rec, lookups)
            if rec["queue_item_id"] == ev["next"]:
                rec["held"] = "next_item"
    res["classified"] = classified
    res["counts"] = summarize(classified)
    res["preserved"] = preserved_families(classified)
    res["retained_reasons"] = retained_program_reasons(classified)
    res["candidates"] = res["counts"][ACT_WOULD_DELETE]
    res["protected_count"] = res["counts"]["music_or_provider"]
    res["watched_count"] = (len(classified) - res["candidates"]
                            - res["protected_count"])
    res["remaining"] = res["candidates"]
    res["more_remaining"] = res["candidates"] > 0
    res["outcome"] = "classified"
    res["reason"] = "" if _units_consistent(ev) else "index_units_mismatch"


def _snapshot_brief(ev: dict[str, Any]) -> dict[str, Any]:
    return {"head": ev["head"], "current_index": ev["current_index"],
            "index_in_buffer": ev["index_in_buffer"], "state": ev["state"],
            "total": ev["total"], "current_position": ev["current_position"],
            "next": ev.get("next"), "future_count": len(ev["future"])}


def _belt(rec: dict[str, Any], lookups) -> str:
    gate = lookups["dj_gate"] if rec["kind"] == KIND_DJ else lookups["news_gate"]
    try:
        return gate_verdict(gate(rec["token"]))
    except (sqlite3.Error, OSError):
        return GATE_UNKNOWN            # an unanswerable gate is not a tombstone


def _mutate(queue_id, res, *, now_dt, db_path, deps, lim, clock, t0,
            ignore_gap=False):
    """Inside the station mutation lock."""
    delete_fn = deps.get("delete_fn") or ma_queue.delete_queue_item

    # Revalidate the exclusions INSIDE the lock: a refill claim written a
    # moment ago is exactly what the lock exists to make visible.
    exclusion = (deps.get("exclusion_fn") or _default_exclusion)(queue_id)
    res["preconditions"]["exclusion"] = exclusion
    if exclusion:
        res["outcome"], res["reason"] = "excluded", exclusion
        return
    if not res["preconditions"]["gap_ok"] and not ignore_gap:
        res["outcome"], res["reason"] = "min_gap", "too_soon_since_last_pass"
        return

    # --- snapshot ----------------------------------------------------------
    ev, why = _read(queue_id, deps, res)
    if ev is None:
        res["outcome"], res["reason"] = "unreadable", why
        return
    res["snapshot"] = _snapshot_brief(ev)
    if ev["state"] not in SUPPORTED_STATES:
        res["outcome"], res["reason"] = "unsupported_state", str(ev["state"])
        return
    buf = ev["index_in_buffer"]
    if not isinstance(buf, int) or isinstance(buf, bool):
        res["outcome"], res["reason"] = "refused", "no_index_in_buffer"
        return
    if not _units_consistent(ev):
        res["outcome"], res["reason"] = "refused", "index_units_mismatch"
        return

    lookups = _lookups(deps, db_path)
    classified = classify_future(
        ev["future"], queue_id=queue_id,
        current_position=ev["current_position"], index_in_buffer=buf,
        lookups=lookups)
    # Autoplay refilling with spent program audio is said, never repaired
    # here: the spent items are removed below like any others
    res["autoplay_program_tail"] = autoplay_watch.observe(
        queue_id, autoplay=ev.get("autoplay"), classified=classified,
        at=now_dt.timestamp(), delete_verdict=V_DELETE)
    res["counts"] = summarize(classified)
    res["preserved"] = preserved_families(classified)
    res["retained_reasons"] = retained_program_reasons(classified)
    candidates = [r for r in classified if r["action"] == ACT_WOULD_DELETE]
    cand_ids = {r["queue_item_id"] for r in candidates}
    # PROTECTED = the customer's music and the provider's picks: every
    # future item with no program route. Their survival, in order, is the
    # invariant the latch guards. WATCHED = every program item we will not
    # remove (live, unknown, buffered): never deleted by us, and reported
    # if something else removes one, but not latch-worthy -- a URL that
    # looks like ours is not evidence of the customer's music either way.
    protected = [r["queue_item_id"] for r in classified if not r["kind"]]
    watched = [r["queue_item_id"] for r in classified
               if r["kind"] and r["queue_item_id"] not in cand_ids]
    res["candidates"] = len(candidates)
    res["protected_count"] = len(protected)
    res["watched_count"] = len(watched)
    res["stale_buffer_held"] = stale_buffer_held(
        classified, ev["current_position"], buf)
    if not candidates:
        if res["stale_buffer_held"]:
            # spent, and held only by MA's stale buffer index: owed, not done
            res["outcome"], res["reason"] = "refused", REASON_STALE_BUFFER
            return
        res["outcome"], res["reason"] = "no_candidates", ""
        return
    # Nearest the head first: the stale item about to be reached is the one
    # the listener would hear next as a silence, so a backlog bigger than
    # one pass is cleared from the head outwards (farthest-first would
    # leave the imminent ones for last). Every
    # confirmed delete moves each farther candidate one place nearer, and
    # the loop says so, so the remaining hints stay exact. The one
    # exception is the item Music Assistant names as next, which is never
    # deleted (the last look, below). MA names the item right after the
    # head all track long, so a stale item there is held on every pass and
    # airs once as the short silence -- the price of never deleting an
    # item the player may already hold.
    candidates.sort(key=lambda r: r["position"])

    head = ev["head"]
    base_index = ev["current_index"]
    outcome, reason = "completed", ""
    abort = None
    stamped = False

    for n, cand in enumerate(candidates):
        if res["attempted"] >= lim.max_deletes:
            outcome, reason = "bounded", "count_limit"
            break
        if clock() - t0 >= lim.time_budget_s:
            outcome, reason = "bounded", "time_budget"
            break
        qid = cand["queue_item_id"]

        # --- cheap revalidation before EVERY delete ---------------------------
        fresh, why = _probe(queue_id, deps, res)
        if fresh is None:
            abort = ("reread_failed", why)
            break
        problem = _revalidate_head(fresh, head, base_index)
        if problem:
            abort = (problem, qid)
            break
        try:
            pos, item = _locate(queue_id, qid, cand["position"], deps, res)
            if pos is None:
                # Out of its window: something nearer the head was inserted
                # or removed, or the item is gone. One complete read settles
                # which, and refreshes every remaining hint.
                full, why = _read(queue_id, deps, res)
                if full is None:
                    abort = ("reread_failed", why)
                    break
                problem = _revalidate_head(full, head, base_index)
                if problem:
                    abort = (problem, qid)
                    break
                _refresh_hints(candidates, full)
                pos = _position_of(full, qid)
                item = (full["future"][pos - full["current_position"] - 1]
                        if pos is not None else None)
        except ma_queue.MAQueueError as exc:
            abort = ("locate_failed", f"{qid}:{str(exc)[:120]}")
            break
        if pos is None:
            res["skipped"].append({"queue_item_id": qid, "reason": "disappeared"})
            continue
        if pos <= fresh["index_in_buffer"]:
            res["skipped"].append({"queue_item_id": qid, "reason": "buffered"})
            continue
        if fresh["next"] == qid:
            # MA's named next item, already visible: skipped here, before
            # it costs a reclassification, the belt or the gap stamp.
            res["skipped"].append({"queue_item_id": qid, "reason": "next_item"})
            continue
        again = classify_item(item, queue_id=queue_id, lookups=lookups)
        if again["verdict"] != V_DELETE or again["token"] != cand["token"]:
            res["skipped"].append({"queue_item_id": qid,
                                   "reason": f"no_longer_stale:{again['reason']}"})
            continue
        belt = _belt(again, lookups)
        if belt != GATE_RETIRED:
            # Lifecycle says stale, the serving authority does not agree.
            # That is a finding, not an obstacle; the pass stops here.
            abort = (f"serving_gate_{belt}", qid)
            break

        # --- the gap stamp, BEFORE the last look --------------------------------
        # Stamped once per pass, by the first candidate to get this far. It
        # is a sqlite commit -- an fsync, and up to the busy timeout behind
        # another writer -- so it may not sit between the last look and the
        # write. A pass that only classified, refused or skipped before this
        # point writes nothing. One whose candidate the last look then skips
        # or stops has stamped without an attempt: rare, and it costs only
        # the gap.
        if not stamped:
            _note_pass(now_dt, db_path)
            stamped = True

        # --- the last look, after every slower step above ---------------------
        # Music Assistant can commit its NEXT item to the player while that
        # item is still beyond index_in_buffer. In MA 2.10, at the preload
        # boundary a few seconds before a track ends, MA can ACCEPT a
        # delete of its committed next item (it refuses one at or below the
        # buffer): the item goes, MA keeps naming the deleted id as next, and
        # the player's request for it is a 404 -- idle seconds early. What MA
        # names as next is read from next_item, the only "next" the queue
        # state gives; MA does not say which reference it keeps, so that is
        # not assumed. The probe at the top of this
        # candidate came before its locate, reclassification, belt and the
        # gap stamp, so it is taken again here. An item MA now names as
        # next, or that now sits in the buffer, is skipped and never sent a
        # delete; a head that moved stops the pass as above.
        last, why = _probe(queue_id, deps, res)
        if last is None:
            abort = ("reread_failed", why)
            break
        problem = _revalidate_head(last, head, base_index)
        if problem:
            abort = (problem, qid)
            break
        if pos <= last["index_in_buffer"]:
            res["skipped"].append({"queue_item_id": qid, "reason": "buffered"})
            continue
        if last["next"] == qid:
            res["skipped"].append({"queue_item_id": qid, "reason": "next_item"})
            continue

        # --- THE ONE WRITE, then prove it ------------------------------------
        res["attempted"] += 1
        try:
            delete_fn(queue_id, qid)
        except ma_queue.MAQueueError as exc:
            abort = ("delete_error", f"{qid}:{str(exc)[:120]}")
            break
        verify, why = _probe(queue_id, deps, res)
        if verify is None:
            abort = ("verify_unreadable", why)
            break
        try:
            still, _ = _locate(queue_id, qid, pos, deps, res)
        except ma_queue.MAQueueError as exc:
            abort = ("verify_unreadable", f"{qid}:{str(exc)[:120]}")
            break
        if still is not None:
            abort = ("delete_refused", qid)     # still there: no retry
            break
        res["deleted"].append(qid)
        for later in candidates[n + 1:]:
            if later["position"] > pos:
                later["position"] -= 1
        if verify["next"] == qid:
            # Accepted and gone from the list, yet still MA's next: the
            # commit landed after the last look's queue state read and
            # before MA acted on the delete, where no read can see it. An
            # accepted delete is not safe playback -- the player will ask
            # for an id that no longer exists. Nothing here can undo that
            # (no transport verb, by design), so the pass stops and says
            # so instead of carrying on. (Seen only if MA's next_item is
            # the reference that kept the deleted id.)
            logger.error("queue_hygiene_next_item_deleted queue=%s pass=%s "
                         "item=%s (Music Assistant still names the deleted "
                         "item as next)", queue_id, res["pass_id"][:8], qid)
            abort = ("next_item_deleted", qid)
            break
        problem = _revalidate_head(verify, head, base_index)
        if problem:
            abort = (f"{problem}_after_delete", qid)
            break

    if abort:
        outcome, reason = "aborted", f"{abort[0]}:{abort[1]}"

    for s in res.get("skipped", ()):
        s_reason = s.get("reason") or "skipped"
        res["retained_reasons"][s_reason] = res["retained_reasons"].get(s_reason, 0) + 1

    # --- closing invariant: only meaningful once something was removed ------
    if res["attempted"]:
        observed = _read_whole_queue(queue_id, deps, res=res)
        if observed is None:
            detail = {"deleted": list(res["deleted"]),
                      "attempted": res["attempted"],
                      "pass_outcome": f"{outcome}:{reason}"}
            trip_latch(LATCH_CLOSING_UNREADABLE, queue_id=queue_id,
                       pass_id=res["pass_id"], detail=detail, now=now_dt,
                       db_path=db_path)
            res["latch_tripped"] = True
            outcome, reason = "latch_tripped", LATCH_CLOSING_UNREADABLE
        else:
            seen = set(observed)
            # A "deleted" id that is still in the queue somewhere was moved,
            # not removed (the window did not see it). Nothing of the
            # customer's is at stake; it is simply not deleted.
            res["unconfirmed"] = [i for i in res["deleted"] if i in seen]
            if res["unconfirmed"]:
                res["deleted"] = [i for i in res["deleted"] if i not in seen]
                outcome = "aborted"
                reason = f"delete_unconfirmed:{res['unconfirmed'][0]}"
            check = check_subsequence(protected, observed)
            if not check["ok"]:
                detail = {
                    "missing": check["missing"][:DIAG_ID_CAP],
                    "missing_count": len(check["missing"]),
                    "reordered": check["reordered"][:DIAG_ID_CAP],
                    "reordered_count": len(check["reordered"]),
                    "protected_count": len(protected),
                    "observed_count": len(observed),
                    "deleted": list(res["deleted"])[:DIAG_ID_CAP],
                    "attempted": res["attempted"],
                    "pass_outcome": f"{outcome}:{reason}",
                    "note": ("Hygiene removed only the ids listed under "
                             "'deleted', none of which is music. A music "
                             "id missing or reordered here was changed by "
                             "something else during the pass (a listener, "
                             "another writer) or the queue was rebuilt; "
                             "inspect before clearing."),
                }
                trip_latch(LATCH_PROTECTED_VIOLATED, queue_id=queue_id,
                           pass_id=res["pass_id"], detail=detail, now=now_dt,
                           db_path=db_path)
                res["latch_tripped"] = True
                outcome, reason = "latch_tripped", LATCH_PROTECTED_VIOLATED
            watch = check_subsequence(watched, observed)
            if not watch["ok"]:
                res["watched_anomalies"] = {
                    "missing": watch["missing"][:DIAG_ID_CAP],
                    "reordered": watch["reordered"][:DIAG_ID_CAP]}
                logger.warning("queue_hygiene_watched_item_changed queue=%s "
                               "pass=%s missing=%d reordered=%d (not the "
                               "customer's music; not latched)", queue_id,
                               res["pass_id"][:8], len(watch["missing"]),
                               len(watch["reordered"]))
    reached = len(res["deleted"]) + len(res["unconfirmed"]) + len(res["skipped"])
    res["remaining"] = max(0, len(candidates) - reached) + len(res["unconfirmed"])
    res["more_remaining"] = res["remaining"] > 0
    res["outcome"], res["reason"] = outcome, reason


def _refresh_hints(candidates: list[dict[str, Any]], full: dict[str, Any]
                   ) -> None:
    """After a complete read, every remaining candidate's position hint is
    made exact again (an absent one keeps its old hint and will be found
    absent by its own locate)."""
    where = {qid: full["current_position"] + 1 + i
             for i, qid in enumerate(full["ids"])}
    for cand in candidates:
        if cand["queue_item_id"] in where:
            cand["position"] = where[cand["queue_item_id"]]


def _revalidate_head(fresh: dict[str, Any], head: str, base_index: Any
                     ) -> str:
    """'' when the queue is still the one we snapshotted, else why not."""
    if fresh["head"] != head:
        return "head_changed"
    cur = fresh.get("current_index")
    if isinstance(base_index, int) and isinstance(cur, int) and cur < base_index:
        return "position_regressed"
    if fresh["state"] not in SUPPORTED_STATES:
        return "unsupported_state"
    if not _units_consistent(fresh):
        return "index_units_mismatch"
    if not isinstance(fresh.get("index_in_buffer"), int) \
            or isinstance(fresh.get("index_in_buffer"), bool):
        return "no_index_in_buffer"
    return ""


def _position_of(ev: dict[str, Any], queue_item_id: str) -> int | None:
    """Absolute list position of a FUTURE item in a complete read, or None."""
    try:
        offset = ev["ids"].index(queue_item_id)
    except ValueError:
        return None
    return ev["current_position"] + 1 + offset


# --- status ---------------------------------------------------------------------------

def status(db_path=None) -> dict[str, Any]:
    """Compact read-only state for a status surface: permission, latch,
    last mutating pass. No queue read."""
    import settings
    permitted = settings.queue_hygiene_permitted()   # never raises
    latch = latch_state(db_path)
    last = _last_pass_at(db_path)
    # armed: automatic triggers exist. Whether a ROUTINE pass may
    # MUTATE is still `permitted` (the setting) together with a clear
    # latch. `mandatory_triggers` names the triggers that reconcile
    # ownership even when `permitted` is false -- a distinct question from
    # both, and one an operator reading a status page must be able to
    # answer without reading this file.
    import queue_hygiene_triggers
    return {"permitted": permitted, "armed": True,
            "triggers": list(queue_hygiene_triggers.TRIGGERS),
            "mandatory_triggers": list(queue_hygiene_triggers.MANDATORY),
            "latch": latch, "last_pass_at": None if last is None else _iso(last),
            "limits": {"max_deletes": MAX_DELETES_PER_PASS,
                       "time_budget_s": PASS_TIME_BUDGET_S,
                       "min_gap_s": MIN_PASS_GAP_S}}
