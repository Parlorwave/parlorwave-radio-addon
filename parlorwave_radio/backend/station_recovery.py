#!/usr/bin/env python3
"""Proving what happened to a station preparation, from the queue itself.

Music Assistant cannot join a SQLite transaction. Start Radio therefore writes
a `preparing` station row, calls replace_future, and then has to answer a
question no local state can answer on its own: DID THAT WRITE LAND? This
module answers it the only way that is honest -- by reading the ordered queue
back and comparing it against the witness captured before the write.

The same matcher serves two callers, which is the point of putting it here:

  * Start Radio, immediately after its own write, still holding the station
    mutation lock;
  * the lifecycle recovery pass, for a `preparing` row stranded by a crash,
    a restart, or an unreadable queue -- possibly hours later, after tracks
    have played and after the DJ or the news have inserted programming.

THREE OUTCOMES, NEVER TWO.

    LANDED      the queue structurally proves the replacement happened
    NOT_LANDED  the queue still matches the pre-write witness exactly
    UNPROVEN    neither -- and that is a real answer, not a failure

UNPROVEN mutates nothing. It does not deactivate the old station, does not
activate the new one, and above all does not re-issue replace_future: an
external write whose outcome is unknown must never be repeated on a guess.
The row stays `preparing` and the next pass looks again.

NO CLOCK DECIDES ANYTHING HERE. Age is not evidence about a write whose
response was lost. A timer may bring us back to look; it may never stand in
for looking.

QUEUE-READ FACTS THIS DEPENDS ON (verified against Music Assistant):
  * player_queues/items defaults to 500 rows -- an explicit limit is required,
    and read_future_items pages;
  * `offset` is a LIST POSITION and `sort_index` has gaps, so NOTHING here
    does arithmetic on sort_index;
  * the current item is anchored by queue_item_id;
  * an incomplete read is not an empty future, and player_queues/all -- not
    an empty items list -- is what says whether a queue exists at all.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import ma_queue
import service_clock
import station_runway
import station_store

logger = logging.getLogger("musicbrain")

LANDED = "landed"
NOT_LANDED = "not_landed"
UNPROVEN = "unproven"

# Page size for the proving read. A station replacement is <= 40 items and the
# interesting region is the future, but the queue may carry hundreds of
# history rows, so this is a page size and not a cap: read_future_items keeps
# paging until a short page proves the end.
READ_PAGE = 500

#: Raw items read BEHIND the head when a proof may need them. An append is
#: anchored on whatever ended the queue before the write, and when nothing
#: was queued that is the song that was PLAYING -- the ordinary case for a
#: Station Programmer start. A song boundary between the write and the read
#: moves that anchor into history, where a future-only read cannot see it
#: and the proof would stay UNPROVEN forever. Callers opt in; legacy reads
#: are unchanged.
PROOF_HISTORY = 40

#: How long a preparation or a pending refill batch may stay GENUINELY
#: UNPROVEN -- read, and judged neither landed nor not-landed -- before it
#: is given up. Without a bound, a start whose pre-write anchor left
#: the queue (replaced by the listener, or rolled past PROOF_HISTORY) is
#: re-read every reaper pass and never resolved, and while it stands every
#: Dial / Request Line / Mode 2 start is refused as "still being
#: confirmed" with the panel's Stop greyed out. Three reaper passes is
#: long enough for a slow Music Assistant to answer and for a transient
#: mismatch to settle; after that, giving up costs at most an unowned run
#: of songs that simply plays, never silence and never a second write.
#: An UNREADABLE queue is not a verdict and never counts towards this.
UNPROVEN_EXPIRY_S = 15 * 60


def unproven_expired(stamp: Any, now: datetime | None = None) -> bool:
    """Has a row stamped `stamp` (station created_at / batch claimed_at)
    been unproven for UNPROVEN_EXPIRY_S? A stamp that cannot be read never
    expires: the conservative answer is the old behaviour."""
    bounds = service_clock.age_bounds(
        str(stamp) if stamp is not None else None,
        now or service_clock.legacy_now())
    if bounds is None:
        return False
    return bounds[0] >= UNPROVEN_EXPIRY_S


def _expire_preparation(station: dict[str, Any], why: str, now, db_path
                        ) -> None:
    """Give up a preparation that stayed unproven past the bound: its
    pending claim is ABANDONED (never confirmed) and the row fails through
    the ordinary failure path. Nothing is written to Music Assistant; the
    songs, if they landed, simply play."""
    sid = station["station_id"]
    batch = station_store.get_pending_batch(sid, db_path=db_path)
    if batch is not None:
        station_store.resolve_batch(batch["batch_id"], station_store.ABANDONED,
                                    resolution="prepare_unproven_expired",
                                    now=now, db_path=db_path)
    station_store.fail_prepared_station(sid, reason="prepare_unproven_expired",
                                        now=now, db_path=db_path)
    logger.warning("station %s preparation unproven for %ds (%s); given up "
                   "-- new starts are allowed again, nothing was written",
                   sid[:8], UNPROVEN_EXPIRY_S, why)

#: How long the reaper waits for an in-flight station write before looking
#: again next interval instead.
SERIALIZED_WAIT_S = 1.0


def future_music_uris(future_items: list[dict[str, Any]]) -> list[str]:
    """The ordered URIs of the FUTURE items that are real music.

    Programming is filtered FIRST and deliberately: DJ breaks and news
    bulletins are inserted at the NEXT position, so between the write and the
    read one can easily appear in the middle of the station's own tracks. A
    matcher that did not drop them would turn a perfectly good preparation
    into UNPROVEN every time the DJ happened to speak."""
    out = []
    for item in future_items:
        kind, _detail = station_runway.classify_item(item)
        if kind == station_runway.MUSIC:
            out.append(station_runway.item_uri(item))
    return out


def _is_suffix(whole: list[str], part: list[str]) -> bool:
    return len(part) <= len(whole) and whole[len(whole) - len(part):] == part


def classify_preparation(station: dict[str, Any], *,
                         current_queue_item_id: str,
                         current_uri: str,
                         future_items: list[dict[str, Any]],
                         ) -> tuple[str, str]:
    """(verdict, reason) for one `preparing` station. PURE.

    LANDED. replace_future replaces the ENTIRE future, so if it landed the
    future's music IS the intended list -- but only until the first track
    boundary. Once a track finishes, intended[0] becomes the current item and
    the future holds intended[1:]. So the test is SUFFIX, not equality, and
    it is paired with an anchor test that says which of those two worlds we
    are in:

        the current item is still the one we captured before the write   (no
        boundary passed yet), OR the current item is itself one of the
        intended URIs (one or more boundaries passed, and what is playing is
        the station we asked for).

    Requiring exact equality would call a perfectly good station UNPROVEN
    simply because the listener heard a song.

    NOT_LANDED. The future's music still matches the pre-write count AND
    hash exactly, so nothing replaced it. This needs the anchor to be
    unchanged too: if a boundary passed, the OLD future also shifted, the hash
    will not match, and the honest answer becomes UNPROVEN rather than a
    guess.

    UNPROVEN. Everything else -- including a preparation with no captured
    witness whose queue no longer resembles either world."""
    intended = [str(u) for u in (station.get("intended_uris") or [])]
    if not intended:
        return UNPROVEN, "no_intended_uris"

    music = future_music_uris(future_items)
    anchor_held = bool(current_queue_item_id) and (
        current_queue_item_id == station.get("pre_current_queue_item_id"))
    current_is_intended = bool(current_uri) and current_uri in intended

    if _is_suffix(intended, music) and (anchor_held or current_is_intended):
        # An empty future only proves anything if what is PLAYING is ours;
        # otherwise "nothing queued" is equally consistent with never having
        # written at all.
        if music or current_is_intended:
            return LANDED, ("boundary_advanced" if current_is_intended
                            else "exact_future_match")

    pre_count = station.get("pre_future_music_count")
    pre_hash = station.get("pre_future_music_hash") or ""
    if (anchor_held and isinstance(pre_count, int) and pre_count >= 0
            and pre_hash and len(music) == pre_count
            and station_store.uri_list_hash(music) == pre_hash):
        return NOT_LANDED, "queue_matches_pre_write_witness"

    return UNPROVEN, "no_structural_proof"


def build_tail_witness(station: dict[str, Any],
                       future_items: list[dict[str, Any]],
                       ) -> tuple[list[dict[str, Any]], bool]:
    """(witness, is_partial) -- the station-owned items still in the future.

    Identity is queue_item_id, which Music Assistant mints per INSERTION, so a
    listener's own manual copy of one of our songs carries a different id and
    can never be mistaken for ours. The URI is stored beside it purely as the
    fallback for the one case ids cannot cover (MA regenerating them on a
    reconnect, still unproven either way), and `last_region` records where we
    last saw the item so a later disappearance can be read as removal rather
    than as MA trimming its own history.

    sort_index is NOT stored: it has gaps, so it is not an
    identity and not a position.

    is_partial is True when the witness had to be truncated at the cap, which
    is what makes station-owned runway a LOWER BOUND rather than an exact
    count. Initial Start Radio writes at most 40 items against a cap of 64, so
    in practice the initial witness is exact."""
    intended = {str(u) for u in (station.get("intended_uris") or [])}
    witness = []
    for item in future_items:
        kind, _ = station_runway.classify_item(item)
        if kind != station_runway.MUSIC:
            continue
        uri = station_runway.item_uri(item)
        if uri not in intended:
            continue
        witness.append({"queue_item_id": item.get("queue_item_id"),
                        "uri": uri, "last_region": "future"})
    partial = len(witness) > station_store.TAIL_CAP
    return witness[:station_store.TAIL_CAP], partial


def read_queue_evidence(queue_id: str, *, state_fn=None, future_fn=None,
                        history: int = 0) -> dict[str, Any] | None:
    """One COMPLETE ordered read of a queue, or None when it cannot be proven.

    Existence comes from player_queues/all, never from an empty item list:
    player_queues/items answers [] for a queue that does not exist exactly as
    it does for one that is empty, so only queue_state() can tell them apart.
    An unreadable queue, a missing current item, or a read that could not be
    completed all return None -- the caller's contract is to change nothing."""
    read_state = state_fn or ma_queue.queue_state
    read_future = future_fn or ma_queue.read_future_items
    try:
        state = read_state(queue_id)
    except Exception as exc:  # noqa: BLE001 - closed-gate boundary
        logger.info("station recovery: queue %s unreadable (%s)", queue_id, exc)
        return None
    current = (state or {}).get("current_item") or {}
    current_id = current.get("queue_item_id")
    if not current_id:
        return None
    try:
        read = read_future(queue_id, current_id,
                           current_index=state.get("current_index"),
                           total_items=state.get("items"), page=READ_PAGE)
    except Exception as exc:  # noqa: BLE001
        logger.info("station recovery: future unreadable on %s (%s)",
                    queue_id, exc)
        return None
    if not read.get("complete"):
        # An incomplete read is not an empty future. Deciding a station's fate
        # from a partial view is exactly the mistake this returns None to
        # avoid.
        logger.info("station recovery: incomplete read on %s (%s)",
                    queue_id, read.get("incomplete_reason"))
        return None
    out = {"state": state, "current_queue_item_id": current_id,
           "current_uri": current.get("uri") or "",
           "future": read["future"]}
    if history > 0:
        out["history"] = _items_behind_head(queue_id, state, current_id,
                                            history)
    return out


def _items_behind_head(queue_id: str, state: dict[str, Any], head_id: str,
                       back: int) -> list[dict[str, Any]]:
    """Up to `back` raw items immediately BEFORE the head, in queue order.
    Best effort: an unreadable window is no history, which can only leave a
    proof UNPROVEN -- never turn it into a verdict."""
    try:
        win = ma_queue.read_window_items(
            queue_id, head_id, current_index=state.get("current_index"),
            back=back, ahead=0)
    except ma_queue.MAQueueError:
        return []
    items = win.get("items") or []
    head = win.get("current_position", 0) - win.get("base_position", 0)
    return list(items[:max(0, head)])


def resolve_preparing(station: dict[str, Any], evidence: dict[str, Any],
                      *, now=None, db_path=None) -> str:
    """Apply the verdict for one `preparing` station. Returns the verdict.

    LANDED activates it and deactivates the predecessor in ONE transaction.
    NOT_LANDED fails the preparation and leaves the old station alone.
    UNPROVEN changes nothing at all."""
    verdict, reason = classify_preparation(
        station,
        current_queue_item_id=evidence["current_queue_item_id"],
        current_uri=evidence["current_uri"],
        future_items=evidence["future"])
    sid = station["station_id"]
    if verdict == LANDED:
        witness, partial = build_tail_witness(station, evidence["future"])
        station_store.activate_prepared_station(
            sid, queue_id=station.get("queue_id") or "", tail=witness,
            tail_is_partial=partial, now=now, db_path=db_path)
        logger.info("station %s activated by recovery (%s, %d witnesses)",
                    sid[:8], reason, len(witness))
    elif verdict == NOT_LANDED:
        station_store.fail_prepared_station(sid, reason="prepare_not_landed",
                                            now=now, db_path=db_path)
        logger.info("station %s preparation did not land (%s)", sid[:8], reason)
    else:
        logger.info("station %s preparation still unproven (%s)",
                    sid[:8], reason)
    return verdict


# --- refill-batch reconciliation ---------------------------------------------

def ordered_queue(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """[history...] + [current] + future, as one ordered list.

    The append anchor may be the last FUTURE item or, when the queue had no
    future at all, the current item itself -- so both live in one sequence
    and position arithmetic never has to special-case an empty tail. When
    the evidence was read with history (read_queue_evidence(history=N)),
    the items behind the head come first, so an anchor a song boundary has
    since carried into history is still found."""
    head = {"queue_item_id": evidence["current_queue_item_id"],
            "media_item": {"uri": evidence.get("current_uri") or ""}}
    return (list(evidence.get("history") or []) + [head]
            + list(evidence.get("future") or []))


def _uri_of(item: dict[str, Any]) -> str:
    media = item.get("media_item")
    return str(media.get("uri") or "") if isinstance(media, dict) else ""


def classify_append(batch: dict[str, Any], evidence: dict[str, Any]
                    ) -> tuple[str, str]:
    """(verdict, reason) for one pending refill batch. PURE.

    On a quiet queue a global item-count delta would be
    enough. A live station is not quiet: a DJ break or the atomic
    news pair can be inserted at the NEXT position between the pre-write
    snapshot and the readback, and a count delta alone would then either miss
    a real append or claim one that never happened. So the proof is
    POSITIONAL and anchored on identity instead.

    The anchor is `pre_tail_queue_item_id` -- whatever sat at the very end of
    the queue immediately before the write. NEXT insertions land BEFORE it;
    only an append lands after it. So:

        LANDED      the anchor is still present, and the items immediately
                    following it are exactly the intended URIs, in order.
        NOT_LANDED  the anchor is still the LAST item -- nothing was appended.
        UNPROVEN    anything else: the anchor is gone (the queue was
                    replaced), or what follows it does not match.

    Deliberately NOT used: sort_index arithmetic of any kind. MA's
    sort_index is strictly increasing but not contiguous -- an append
    can produce ...2, 4, 5 -- so `old_tail + 1` would fail on an
    ordinary append. Position comes from list
    order and identity from queue_item_id.

    Trailing items beyond our run are tolerated (another client may have
    appended after us); what matters is that OUR run starts immediately at
    the anchor and matches exactly."""
    intended = [str(u) for u in (batch.get("intended_uris") or [])]
    if not intended:
        return UNPROVEN, "no_intended_uris"
    anchor = str(batch.get("pre_tail_queue_item_id") or "")
    if not anchor:
        return UNPROVEN, "no_pre_write_anchor"

    items = ordered_queue(evidence)
    position = next((i for i, it in enumerate(items)
                     if it.get("queue_item_id") == anchor), None)
    if position is None:
        # The item that ended the queue before the write is gone entirely --
        # the queue was replaced or rebuilt. Nothing here can be proven, and
        # guessing is exactly what must not happen.
        return UNPROVEN, "pre_write_tail_absent"

    following = items[position + 1:]
    if not following:
        return NOT_LANDED, "queue_still_ends_at_the_pre_write_tail"
    run = [_uri_of(it) for it in following[:len(intended)]]
    if run == intended:
        extra = len(following) - len(intended)
        return LANDED, ("exact_tail_run" if extra == 0
                        else f"tail_run_with_{extra}_later_items")
    return UNPROVEN, "tail_does_not_match_intended_batch"


def appended_items(batch: dict[str, Any], evidence: dict[str, Any]
                   ) -> list[dict[str, Any]]:
    """The queue items this batch actually created, in order.

    Called only after LANDED, so the run is known to be there. Music
    Assistant mints a fresh queue_item_id per insertion, which is what makes
    these safely distinguishable from a listener's own copy of the same
    song."""
    intended = [str(u) for u in (batch.get("intended_uris") or [])]
    items = ordered_queue(evidence)
    anchor = str(batch.get("pre_tail_queue_item_id") or "")
    position = next((i for i, it in enumerate(items)
                     if it.get("queue_item_id") == anchor), None)
    if position is None:
        return []
    return items[position + 1:position + 1 + len(intended)]


def resolve_pending_batch(batch: dict[str, Any], evidence: dict[str, Any],
                          *, confirm_fn=None, now=None, db_path=None
                          ) -> tuple[str, str]:
    """Apply the verdict for one pending batch. Returns (verdict, reason).

    NEVER appends. A batch whose fate is unknown stays pending and is looked
    at again later; the one thing that must not happen is a second write of
    the same songs."""
    verdict, reason = classify_append(batch, evidence)
    if verdict == LANDED and confirm_fn is not None:
        confirm_fn(batch, evidence)
    elif verdict == NOT_LANDED:
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.NOT_LANDED,
                                    resolution=reason, now=now,
                                    db_path=db_path)
    return verdict, reason


def recover_pass_serialized(*, now=None, db_path=None) -> dict[str, Any]:
    """recover_pass, run only while NO station-owned write is in flight.

    Recovery judges an append by reading the queue. Read in the middle of a
    live start or refill -- after its claim, before its songs land -- it
    would call a write that is about to land NOT_LANDED and fail the very
    claim the writer is about to prove, leaving songs nobody owns and a
    runway that is topped up again. So the reaper, and a start clearing a
    stranded preparation, take the station mutation lock first; a busy lock
    simply means "look again next time". (Start Radio already calls
    recover_pass from inside the lock.)"""
    import station_lock
    try:
        with station_lock.station_mutation(timeout=SERIALIZED_WAIT_S):
            return recover_pass(now=now, db_path=db_path)
    except station_lock.StationMutationBusy:
        return {"checked": 0, "verdict": None, "station_id": None,
                "batch_checked": 0, "batch_verdict": None, "batch_id": None,
                "skipped": "station_mutation_in_flight"}


def recover_pass(*, now=None, db_path=None, state_fn=None, future_fn=None
                 ) -> dict[str, Any]:
    """Reconcile a stranded `preparing` station. Safe to call repeatedly.

    Issues READS only. There is deliberately no path in this function that can
    write to Music Assistant: a preparation whose outcome is unknown is
    resolved by looking, or not at all."""
    result = {"checked": 0, "verdict": None, "station_id": None,
              "batch_checked": 0, "batch_verdict": None, "batch_id": None}
    _recover_pending_batches(result, now=now, db_path=db_path,
                             state_fn=state_fn, future_fn=future_fn)
    try:
        station = station_store.get_preparing_station(db_path=db_path)
    except Exception:
        logger.warning("station recovery: preparing lookup failed",
                       exc_info=True)
        return result
    if station is None:
        return result
    result["checked"] = 1
    result["station_id"] = station["station_id"]
    queue_id = station.get("queue_id") or ""
    if not queue_id:
        # A preparation that never got as far as binding an output cannot be
        # proven against any queue, and never wrote one either.
        station_store.fail_prepared_station(
            station["station_id"], reason="prepare_unbound", now=now,
            db_path=db_path)
        result["verdict"] = NOT_LANDED
        return result
    evidence = read_queue_evidence(queue_id, state_fn=state_fn,
                                   future_fn=future_fn)
    if evidence is None:
        result["verdict"] = UNPROVEN
        return result
    if station_store.is_programmer(station):
        # A Station Programmer start APPENDS its opening runway under its own
        # pending claim, so it is proven like a refill (anchored on the
        # pre-write tail), not like a replaced future. Same rule: read-only.
        import station_live
        result["verdict"] = station_live.resolve_preparation(
            station, evidence, now=now, db_path=db_path)
    else:
        result["verdict"] = resolve_preparing(station, evidence, now=now,
                                              db_path=db_path)
    if (result["verdict"] == UNPROVEN
            and unproven_expired(station.get("created_at"), now)):
        # Read, judged, and still unprovable after the bound: give up, so a
        # stranded start cannot refuse every new start forever.
        _expire_preparation(station, "read verdict unproven", now, db_path)
        result["expired"] = True
    return result


def _recover_pending_batches(result: dict[str, Any], *, now=None, db_path=None,
                             state_fn=None, future_fn=None) -> None:
    """Reconcile a refill batch stranded by a crash or a lost response.

    This is the half of restart recovery that exists because Music Assistant
    answers an append with HTTP 200 and a body of `null`. A process that died
    between the write and the confirmation left a pending row and no local
    evidence whatsoever -- so the queue is read and the batch is judged from
    what is actually there.

    READ-ONLY against Music Assistant, structurally: the only write path for
    a confirmed batch is the local confirmation transaction, and there is no
    branch here that can call append. Startup must never append; a timer may
    bring us here to look, but looking is the only thing that decides."""
    active = None
    try:
        active = station_store.get_active_station(db_path=db_path)
    except Exception:
        logger.warning("refill recovery: active station unreadable",
                       exc_info=True)
        return
    if active is None:
        return
    try:
        batch = station_store.get_pending_batch(active["station_id"],
                                                db_path=db_path)
    except Exception:
        logger.warning("refill recovery: pending batch unreadable",
                       exc_info=True)
        return
    if batch is None:
        return
    result["batch_checked"] = 1
    result["batch_id"] = batch["batch_id"]
    evidence = read_queue_evidence(
        batch.get("queue_id") or "", state_fn=state_fn, future_fn=future_fn,
        history=PROOF_HISTORY if station_store.is_programmer(active) else 0)
    if evidence is None:
        result["batch_verdict"] = UNPROVEN
        return
    import station_refill
    verdict, reason = resolve_pending_batch(
        batch, evidence,
        confirm_fn=lambda b, e: station_refill.confirm_landed_batch(
            active, b, e, now=now, db_path=db_path),
        now=now, db_path=db_path)
    result["batch_verdict"] = verdict
    logger.info("refill batch %s reconciled at startup: %s (%s)",
                batch["batch_id"][:8], verdict, reason)
    if verdict == UNPROVEN and unproven_expired(batch.get("claimed_at"), now):
        # Read, judged, still unprovable after the bound: ABANDON (never
        # confirm) so the station can refill again. Worst case is an unowned
        # run of songs already in the queue plus a fresh refill -- more
        # runway, never silence, never a second write of the same songs.
        station_store.resolve_batch(batch["batch_id"], station_store.ABANDONED,
                                    resolution="refill_unproven_expired",
                                    now=now, db_path=db_path)
        result["batch_expired"] = True
        logger.warning("refill batch %s unproven for %ds (%s); abandoned so "
                       "the station may refill again, nothing was written",
                       batch["batch_id"][:8], UNPROVEN_EXPIRY_S, reason)
