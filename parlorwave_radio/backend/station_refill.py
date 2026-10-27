#!/usr/bin/env python3
"""One station refill, end to end -- the transaction, not the trigger.

Two entry points share one transaction. `replenish()` is the explicit
`POST /station/replenish` command -- unchanged, and deliberately never gated
on Keep Radio Playing, because asking for music by hand is not automatic
behaviour. `replenish_automatic()` is driven by the eligibility policy, honours
the customer switch right up to the final pre-write revalidation, and separates
how many candidates to select from how many to actually append.

Neither duplicates the machinery: the claim, the single external write, the
readback and the reconciliation are the same code for both, because one proven
path is safer than two.

THE SHAPE OF THE PROBLEM. Music Assistant cannot join a SQLite transaction,
and its append answers HTTP 200 with a body of `null` -- no ids, no echo. So
between "we decided to append" and "we know what happened" there is a window
no local state can describe, and the only way through it is to write down the
intent first and prove the outcome afterwards from the queue itself:

    select (pure, expensive)  ->  resolve to URIs  ->  fresh pre-write witness
    ->  persist the claim     ->  [ lock: revalidate -> append -> read back
                                   -> reconcile -> confirm ]

The claim is created AFTER selection on purpose. Selection and URI resolution
take seconds -- resolution is one serial Music Assistant search per pick --
and a claim held across them would protect nothing while making a legitimate
slow batch look abandoned. Claim what you are about to write, not what you
are about to think about.

WHAT NEVER HAPPENS HERE. A second append in one invocation, under any
outcome. A retry of a write whose result is unknown. A queue mutation from
recovery. A station deactivated because a refill failed. If any load-bearing
proof is missing, the batch stays pending, nothing is written, and the music
that is already queued keeps playing -- which is always a better answer than
a duplicate.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

import media_ref
import service_clock
import station_lock
import station_recovery
import station_runway
import station_selector
import station_store

logger = logging.getLogger("musicbrain")

DEFAULT_COUNT = 12


class _NotReorderable(Exception):
    """Internal sentinel: the candidate set is provider-
    authoritative music, so ParlorWave reordering (LLM programmer /
    same-artist policy) is
    skipped by design -- not a failure."""
MIN_COUNT, MAX_COUNT = 1, 12

# How much play history the selector's rotation windows can see. The deepest
# window it uses is "the last 40 identities", so this is that with room to
# spare -- not a lifetime history, which the selector is explicitly not given.
RECENT_HISTORY = 120


# --- inputs -------------------------------------------------------------------

def _recent_played(session_id: str | None, db_path=None
                   ) -> list[dict[str, Any]]:
    """Newest-first play history for the selector's rotation windows.

    Read from `session_tracks`, which is where playback actually records what
    was heard; the selector is pure and must never query anything itself.
    Programming is excluded at the source by `media_ref.programming_ref`, so
    a DJ break can never occupy a rotation slot meant for music."""
    import playback
    c = playback._connect(db_path)
    try:
        rows = c.execute(
            "SELECT path, artist, album, media_uri, started_at "
            "FROM session_tracks WHERE started_at IS NOT NULL "
            "ORDER BY started_at DESC LIMIT ?", (RECENT_HISTORY,)).fetchall()
    finally:
        c.close()
    out = []
    for path, artist, album, uri, started in rows:
        if media_ref.programming_ref(uri)[0]:
            continue
        if not path:
            continue
        out.append({"path": path, "artist": artist or "",
                    "album": album or "", "played_at": started})
    return out


def _recently_generated(station: dict[str, Any]) -> list[dict[str, Any]]:
    """The station's rotation ring, in the selector's identity space.

    Older entries carry only {uri, at} -- the selector
    keys on PATH, so those older entries simply contribute nothing rather
    than being reinterpreted. A URI is not a path and pretending otherwise
    would silently exclude the wrong tracks. No migration: the column is
    JSON, and new entries carry both."""
    out = []
    for row in station.get("generated") or ():
        if isinstance(row, dict) and row.get("path"):
            out.append({"path": row["path"], "at": row.get("at")})
    return out


def _queued_identities(evidence: dict[str, Any], catalog_tracks: list
                       ) -> set[str]:
    """Catalog paths for everything currently in the queue -- current item
    included -- so a refill can never propose a song the listener is about to
    hear anyway.

    The queue speaks URIs and the selector speaks paths, so this maps across
    using the catalog itself rather than guessing."""
    by_uri: dict[str, str] = {}
    for track in catalog_tracks or ():
        uri = str(track.get("media_uri") or "")
        if uri:
            by_uri[uri] = str(track.get("path") or "")
    uris = {evidence.get("current_uri") or ""}
    for item in evidence.get("future") or ():
        uris.add(station_runway.item_uri(item))
    return {by_uri[u] for u in uris if u and u in by_uri}


def _resolve(picks: list[dict[str, Any]], search_fn) -> list[dict[str, Any]]:
    """Catalog picks -> [{path, uri, artist, title}], dropping the
    unresolvable.

    The PATH IS CARRIED THROUGH RESOLUTION, which the initial Start Radio
    path never needed: the rotation ring stores both, and losing the path
    here would mean the very tracks this refill just queued could be chosen
    again by the next one.

    Deliberately the existing sequential title-search-plus-exact-match
    behaviour, and deliberately NOT an artist/title cache: that identity
    merges the remasters, live cuts and duplicate rows the library genuinely
    holds, which is exactly what path exists to keep apart. Slower and
    correct beats faster and wrong here."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for track in picks:
        title, artist = track.get("title"), track.get("artist")
        path = str(track.get("path") or "")
        if not (title and artist and path):
            continue
        try:
            hits = search_fn(title, 8)
        except Exception:
            continue
        for hit in hits or ():
            name = str(hit.get("name") or "").strip().casefold()
            artists = [str(a).strip().casefold() for a in hit.get("artists") or []]
            uri = hit.get("uri")
            if (name == title.strip().casefold()
                    and artist.strip().casefold() in artists
                    and uri and uri not in seen):
                out.append({"path": path, "uri": uri,
                            "artist": artist, "title": title})
                seen.add(uri)
                break
    return out


# --- witness ------------------------------------------------------------------

def build_witness(station: dict[str, Any], evidence: dict[str, Any],
                  appended: list[dict[str, Any]]) -> tuple[list, bool]:
    """The station's ownership witness after a proven append.

    Surviving FUTURE witnesses are kept, consumed ones are dropped, and the
    newly appended items are added. Pruning matters: a witness that only ever
    grew would drift into a lifetime play log, and the thing it is supposed
    to answer is "what does this station still own", not "what did it once
    queue".

    Entries that have moved PAST are dropped rather than remembered as gone,
    because Music Assistant trims its own history and a pruned entry must
    never later read as evidence that a listener removed something."""
    live_ids = {it.get("queue_item_id") for it in evidence.get("future") or ()}
    kept = [w for w in station.get("tail") or ()
            if w.get("queue_item_id") in live_ids]
    for item in appended:
        if item.get("queue_item_id") not in live_ids:
            # Proven across a song boundary (history-aware evidence): a
            # batch song already playing or played is not runway.
            continue
        kept.append({"queue_item_id": item.get("queue_item_id"),
                     "uri": station_runway.item_uri(item),
                     "last_region": "future"})
    partial = len(kept) > station_store.TAIL_CAP
    return kept[:station_store.TAIL_CAP], partial


def confirm_landed_batch(station: dict[str, Any], batch: dict[str, Any],
                         evidence: dict[str, Any], *, now=None,
                         db_path=None) -> bool:
    """The confirmation transaction, shared by the live request and startup
    recovery so both land a proven batch identically."""
    appended = station_recovery.appended_items(batch, evidence)
    witness, partial = build_witness(station, evidence, appended)
    stamp = (now or service_clock.legacy_now()).replace(microsecond=0).isoformat()
    by_uri = {str(m.get("uri")): str(m.get("path") or "")
              for m in (batch.get("intended_map") or [])}
    # A Station Programmer station's selector identity IS the library URI,
    # so a batch confirmed by startup recovery (which never had the
    # in-memory intended_map) still lands in the rotation ring under the
    # identity the next refill will exclude. A legacy station keys on a
    # catalog path, which a URI is not -- that fallback stays empty.
    uri_is_path = station_store.is_programmer(station)
    ring = list(station.get("generated") or ())
    anchors = batch.get("anchor_paths") or ()
    for item in appended:
        uri = station_runway.item_uri(item)
        entry = {"uri": uri,
                 "path": by_uri.get(uri, "") or (uri if uri_is_path else ""),
                 "at": stamp}
        if entry["path"] and entry["path"] in anchors:
            # Lightweight programming memory: this entry was picked AS a
            # recognisable anchor. Diagnostic provenance for the ring;
            # nothing keys on its absence (older entries never carry it).
            entry["anchor"] = True
        ring.append(entry)
    return station_store.confirm_batch_and_witness(
        batch["batch_id"], station["station_id"], tail=witness,
        tail_is_partial=partial, generated=ring, now=now, db_path=db_path)


def _programmer_order(tracks: list[dict[str, Any]], features, feel: str,
                      programmer_fn) -> list[dict[str, Any]]:
    """The LLM-programmer seam: an optional advisor may
    REORDER the deterministic selection. Nothing more.

    Contract, deliberately narrow because a model can be slow and can answer
    in the wrong shape (a JSON object where an id array was requested):

      * `programmer_fn(tracks, features, feel)` returns an iterable of
        track identities (station_selector.track_identity) in play order.
      * Returned ids are validated against the selection: unknown ids and
        duplicates are dropped; ids it did not mention keep their
        deterministic order AFTER the ones it did.
      * The advisor can therefore never add a track, never drop one, never
        shrink a batch, and never slow the refill beyond its own timeout --
        exceptions and garbage leave the deterministic order untouched.

    The optional LLM radio programmer plugs a hardened caller in here;
    Endless Radio must keep working when it is absent, slow, or wrong,
    which this contract guarantees by construction."""
    if programmer_fn is None or len(tracks) < 2:
        return tracks
    by_id = {station_selector.track_identity(t): t for t in tracks}
    try:
        proposed = programmer_fn(list(tracks), features, feel)
        ordered, seen = [], set()
        for ident in proposed or ():
            ident = str(ident)
            if ident in by_id and ident not in seen:
                ordered.append(by_id[ident])
                seen.add(ident)
        if not ordered:
            return tracks
        ordered.extend(t for t in tracks
                       if station_selector.track_identity(t) not in seen)
        return ordered
    except Exception:
        logger.warning("station programmer advisor failed "
                       "(deterministic order keeps)", exc_info=True)
        return tracks



# --- the engine ---------------------------------------------------------------

def _ownership(station: dict[str, Any], evidence: dict[str, Any],
               automatic: bool, now_dt: datetime, db_path,
               read_fn=None, write_fn=None) -> dict[str, Any]:
    """Fenced ownership step. A failure here reports and allows: the
    one-writer proof is never worth a silent station."""
    try:
        import queue_ownership
        flag = (evidence.get("state") or {}).get("autoplay_enabled")
        observed = flag if isinstance(flag, bool) else None
        if automatic:
            return queue_ownership.enforce(
                station=station, observed=observed, now=now_dt,
                db_path=db_path, read_fn=read_fn, write_fn=write_fn,
                source="refill")
        return queue_ownership.reassert(
            station=station, observed=observed, now=now_dt, db_path=db_path,
            read_fn=read_fn, write_fn=write_fn)
    except Exception:
        logger.warning("queue ownership step failed (refill proceeds)",
                       exc_info=True)
        return {"state": "unverified", "refill_allowed": True,
                "error": "ownership_step_failed"}


def _keep_radio_playing(settings_fn=None) -> bool:
    """The customer switch, read fresh. Unreadable settings must never take a
    decision down, so the shipped default applies."""
    try:
        if settings_fn is not None:
            return bool(settings_fn())
        import settings
        return bool(settings.load()["playback"]["keep_radio_playing"])
    except Exception:
        logger.warning("keep_radio_playing unreadable; assuming enabled",
                       exc_info=True)
        return True


def _fail(reason: str, **extra) -> dict[str, Any]:
    out = {"ok": False, "reason": reason, "appended_count": 0}
    out.update(extra)
    return out


def replenish(count: int = DEFAULT_COUNT, **kw) -> dict[str, Any]:
    """MANUAL replenishment -- the explicit user/diagnostic command.

    Unchanged: `count` is both how many to select and the most to append, and
    the customer's Keep Radio Playing switch is deliberately NOT consulted.
    Asking for music by hand is not automatic behaviour, so it must keep
    working with automation turned off."""
    count = max(MIN_COUNT, min(MAX_COUNT, int(count)))
    return _replenish(select_count=count, append_count=count,
                      automatic=False, **kw)


def replenish_automatic(*, desired_append_count: int, selection_budget: int,
                        **kw) -> dict[str, Any]:
    """AUTOMATIC replenishment, driven by the eligibility policy.

    The two counts are deliberately different and must not be collapsed.
    Music Assistant's exact artist+title matcher refuses to guess, so picks
    routinely fail to resolve -- only some of a selection may resolve.
    `selection_budget` therefore carries a modest surplus so an ordinary miss
    does not produce a short batch, while `desired_append_count` stays the
    hard ceiling on what is actually written.

    Surplus that resolves beyond the ceiling is DISCARDED, not appended and
    not remembered: it never enters the pending batch, the pre-write witness,
    the ownership witness or the rotation ring. Only music that actually
    reached the queue is recorded as having been queued.

    Automatic mode additionally re-checks Keep Radio Playing at the final
    pre-write revalidation."""
    desired = max(0, min(MAX_COUNT, int(desired_append_count)))
    budget = max(desired, min(MAX_COUNT, int(selection_budget)))
    if desired <= 0:
        return _fail("nothing_desired")
    return _replenish(select_count=budget, append_count=desired,
                      automatic=True, **kw)


def _replenish(*, select_count: int, append_count: int, automatic: bool,
               now: datetime | None = None,
               db_path=None, state_fn=None, future_fn=None, search_fn=None,
               append_fn=None, library_fn=None, rng=None,
               settings_fn=None, feel_fn=None, features_fn=None,
               programmer_fn=None, autoplay_read_fn=None,
               autoplay_write_fn=None) -> dict[str, Any]:
    """The one transaction, shared by both entry points. At most ONE append,
    ever. Every refusal leaves the queue exactly as it was; there is no path
    from a failed precondition to a Music Assistant write."""
    import ma_queue
    now_dt = now or service_clock.legacy_now()
    count = select_count

    station = station_store.get_active_station(db_path=db_path)
    if station is None:
        return _fail("no_active_station")
    sid, queue_id = station["station_id"], station.get("queue_id") or ""
    if station_store.is_programmer(station):
        # A Station Programmer station is refilled by station_live from its
        # StationIntent and Music Assistant's library -- never from the
        # local catalog by title search. One station, one refill engine.
        return _fail("station_programmer_owns_refill", station_id=sid,
                     queue_id=queue_id)
    weights = station.get("artist_weights") or {}
    if not weights or not queue_id:
        return _fail("station_profile_unusable", station_id=sid)

    read = dict(state_fn=state_fn, future_fn=future_fn)
    evidence = station_recovery.read_queue_evidence(queue_id, **read)
    if evidence is None:
        return _fail("queue_unreadable_or_incomplete",
                     station_id=sid, queue_id=queue_id)
    if (evidence["state"] or {}).get("state") != "playing":
        return _fail("not_playing", station_id=sid, queue_id=queue_id)
    if media_ref.programming_ref(evidence["current_uri"])[0]:
        # Fail closed while a DJ break or a bulletin is on air: the tail append
        # itself would be safe, but a refusal costs only a retry.
        return _fail("programming_on_air", station_id=sid, queue_id=queue_id)

    # An unresolved claim is reconciled FIRST, and settles this request. One
    # HTTP call performs at most one append, so a request that had to clean
    # up after a previous one does not also start a new batch.
    pending = station_store.get_pending_batch(sid, db_path=db_path)
    if pending is not None:
        verdict, reason = station_recovery.resolve_pending_batch(
            pending, evidence,
            confirm_fn=lambda b, e: confirm_landed_batch(
                station, b, e, now=now_dt, db_path=db_path),
            now=now_dt, db_path=db_path)
        return {"ok": verdict != station_recovery.UNPROVEN,
                "reason": "reconciled_prior_batch", "appended_count": 0,
                "station_id": sid, "queue_id": queue_id,
                "batch_id": pending["batch_id"], "batch_state": verdict,
                "reconciliation": reason}

    # Queue ownership. ParlorWave is about to program this queue, so Music
    # Assistant's own Autoplay must be off for it -- checked from the flag the
    # evidence snapshot already carries, AFTER reconciliation (cleaning up a
    # prior claim programs nothing) and BEFORE anything is selected. Automatic
    # refill YIELDS while the queue is contested (one autonomous programmer,
    # and it is not ParlorWave until the flag is observed off again); an
    # explicit person-issued replenish is the listener choosing ParlorWave,
    # which re-asserts ownership and resets the drift budget.
    ownership = _ownership(station, evidence, automatic, now_dt, db_path,
                           autoplay_read_fn, autoplay_write_fn)
    if automatic and not ownership.get("refill_allowed", True):
        return _fail("ma_autoplay_contested", station_id=sid,
                     queue_id=queue_id, ownership=ownership)

    try:
        import library_repo
        catalog_tracks = (library_fn or library_repo.get_library)()["tracks"]
    except Exception:
        return _fail("library_unavailable", station_id=sid)
    from musicdb import get_scores_for_artists
    scores = get_scores_for_artists(list(weights))

    # Station feel (Studio) is read FRESH per refill and reaches selection
    # only as the RefillPolicy: the frozen roster proposes exactly as before,
    # the feel decides rotation, spacing and how hard the weights are
    # followed. Nothing already queued is touched by a feel change.
    import station_feel
    feel = station_feel.current_feel(
        None if feel_fn is None else {"station": {"feel": feel_fn()}})

    # Programming features: personal familiarity per track from play
    # history, one grouped query. Advice for the feel's policy, never a
    # gate -- any failure selects exactly as if no features existed.
    features = None
    try:
        import station_features
        if features_fn is not None:
            features = features_fn()
        else:
            features = station_features.features_for(
                catalog_tracks,
                stats_by_path=station_features.play_stats(list(weights)),
                artist_weights=weights)
    except Exception:
        logger.warning("station features unavailable (refill proceeds)",
                       exc_info=True)
        features = None

    selection = station_selector.select_station_refill(
        artist_weights=weights, request_text=station.get("request_text") or "",
        catalog_tracks=catalog_tracks, scores=scores,
        rng=rng or random.Random(), count=count, now=now_dt,
        hard_excluded_ids=_queued_identities(evidence, catalog_tracks),
        recent_played=_recent_played(None, db_path=db_path),
        recently_generated=_recently_generated(station),
        policy=station_feel.policy_for(feel), features=features)
    if not selection.tracks:
        return _fail("no_candidates", station_id=sid, queue_id=queue_id,
                     requested_count=count, selected_count=0, feel=feel,
                     relaxation_pass=selection.relaxation_pass)

    # The optional LLM radio programmer: reorder-only, deadline-
    # yielding, failure-safe. Runs HERE -- the unserialized selection
    # phase, before resolution and far outside the station mutation lock --
    # so a slow model can never hold a lock, an observer or the ingestion
    # thread. An injected programmer_fn (tests, future advisors) bypasses
    # the production advisor entirely.
    # LLM-programmer / artist-policy HARD BOUNDARY (Pandora). The programmer
    # and same-artist policy reorder ParlorWave's OWN local candidate set
    # only. Provider-authoritative
    # music (Pandora dynamic radio) is NEVER a reorderable candidate: this
    # selection is drawn from the local catalog and must contain no
    # provider-owned item, so if one ever appeared we skip reordering entirely
    # rather than let ParlorWave touch a provider's chosen order. Pinned by
    # test.
    import source_capabilities as _srccap
    _reorderable = all(
        _srccap.item_provenance(t) != _srccap.PANDORA_PROVIDER_MUSIC
        for t in selection.tracks)
    prog_telemetry = None
    advisor = programmer_fn
    if advisor is None and _reorderable:
        import station_programmer
        by_path = {str(t.get("path") or ""): t for t in catalog_tracks}
        recent = []
        for row in reversed(_recent_played(None, db_path=db_path)[:8]):
            hit = by_path.get(row.get("path"))
            title = (hit or {}).get("title") or ""
            name = f"{row.get('artist') or '?'} - {title}" if title \
                else (row.get("artist") or "?")
            recent.append(name)
        outcome = station_programmer.reorder(
            selection.tracks, features, feel,
            station_desc=station.get("request_text") or "",
            recent=recent, queue_id=queue_id, now=now_dt, db_path=db_path)
        prog_telemetry = outcome["telemetry"]
        ordered = outcome["ordered"]
        if ordered is not None:
            advisor = lambda t, f, fl: ordered  # noqa: E731 - validated ids
    if _reorderable:
        selection.tracks = _programmer_order(selection.tracks, features, feel,
                                             advisor)

    # Same-artist policy: the FINAL deterministic say on artist
    # adjacency, after the LLM reorder. Governs only this autonomous batch;
    # returns an exact permutation. The LLM proposes; this enforces. Never runs
    # on provider-authoritative music (the _reorderable boundary above).
    artist_telemetry = None
    try:
        if not _reorderable:
            raise _NotReorderable       # provider order is never repaired
        import artist_policy
        import settings as _sset
        sa = (_sset.load().get("station") or {}).get("same_artist") or {}
        outcome = artist_policy.enforce(
            selection.tracks,
            enabled=bool(sa.get("enabled", True)),
            frequency=sa.get("frequency", "medium"))
        selection.tracks = outcome["order"]
        artist_telemetry = outcome["telemetry"]
    except _NotReorderable:
        pass                            # Pandora boundary: quiet, not a failure
    except Exception:
        logger.warning("artist-adjacency policy failed (order unchanged)",
                       exc_info=True)


    resolved = _resolve(selection.tracks,
                        search_fn or ma_queue.search_library_tracks)
    resolved_total = len(resolved)
    # THE CEILING. Surplus exists only to absorb resolution misses; anything
    # beyond what we intend to append is dropped here, BEFORE the claim, so no
    # durable record ever mentions a track that was not queued.
    resolved = resolved[:append_count]
    if not resolved:
        return _fail("resolve_failed", station_id=sid, queue_id=queue_id,
                     requested_count=append_count,
                     selected_count=len(selection.tracks), resolved_count=0,
                     relaxation_pass=selection.relaxation_pass)
    anchor_paths = {str(t.get("path") or "")
                    for t in selection.tracks
                    if station_selector.track_identity(t)
                    in set(selection.anchor_ids)}
    base = {"station_id": sid, "queue_id": queue_id,
            "requested_count": append_count,
            "selection_budget": select_count,
            "selected_count": len(selection.tracks),
            "resolved_count": resolved_total,
            "automatic": automatic,
            "relaxation_pass": selection.relaxation_pass, "feel": feel,
            "programmer": prog_telemetry,
            "artist_policy": artist_telemetry,
            "ownership": ownership,
            "selected_tracks": [{"artist": t.get("artist"),
                                 "title": t.get("title")}
                                for t in selection.tracks]}
    return commit_append(station, queue_id, resolved, base=base,
                         automatic=automatic, now_dt=now_dt, db_path=db_path,
                         read=read, append_fn=append_fn,
                         settings_fn=settings_fn, anchor_paths=anchor_paths)


def commit_append(station: dict[str, Any], queue_id: str,
                  resolved: list[dict[str, Any]], *, base: dict[str, Any],
                  automatic: bool, now_dt: datetime, db_path=None,
                  read: dict[str, Any] | None = None, append_fn=None,
                  settings_fn=None, honour_switch: bool = True,
                  settle_fn=None,
                  anchor_paths=()) -> dict[str, Any]:
    """THE ONE APPEND, shared by every station refill -- legacy Endless
    Radio and the Station Programmer (station_live) alike:

        fresh pre-write witness -> persist the claim -> [ lock: revalidate
        -> append -> read back -> reconcile -> confirm ]

    `resolved` is [{path, uri, artist, title}] -- exact library URIs, in
    play order, already capped to what is to be written. An AUTOMATIC
    append re-reads the customer's Keep Radio Playing switch inside the
    lock; `honour_switch=False` is the Station Programmer's, whose station
    a person started explicitly and stops explicitly (Stop Station), so the
    switch for legacy automatic refill does not govern it. `settle_fn`, when
    given, is called once if the append RAISED and has not landed at the
    first read, and the queue is read again before the verdict -- a lost
    response can still be applied late. The caller owns that wait (this
    engine has no timer); legacy refill passes none and keeps its immediate
    verdict. `base` is merged into every answer.

    At most ONE external write under any outcome; every refusal leaves the
    queue exactly as it was. The revalidation inside the lock is what stops
    a batch belonging to a station that was stopped, switched or superseded
    while its tracks were being chosen: the station_id is its generation."""
    sid = station["station_id"]
    read = dict(read or {})
    # A Station Programmer's proofs read history too, so an append anchored
    # on the playing song is still provable after a boundary.
    if station_store.is_programmer(station):
        read["history"] = station_recovery.PROOF_HISTORY
    uris = [r["uri"] for r in resolved]

    # A FRESH complete read for the witness: selection took real time, and a
    # stale snapshot would anchor the proof to a tail that has since moved.
    evidence = station_recovery.read_queue_evidence(queue_id, **read)
    if evidence is None:
        return _fail("queue_unreadable_or_incomplete", station_id=sid,
                     queue_id=queue_id)
    ordered = station_recovery.ordered_queue(evidence)
    pre_tail = ordered[-1]
    batch = station_store.create_pending_batch(
        station_id=sid, queue_id=queue_id, intended_uris=uris,
        pre_items_count=(evidence["state"] or {}).get("items") or len(ordered),
        pre_tail_queue_item_id=str(pre_tail.get("queue_item_id") or ""),
        pre_tail_position=len(ordered) - 1,
        pre_qiid_hash=station_store.uri_list_hash(
            [str(i.get("queue_item_id") or "") for i in ordered]),
        pre_uri_multiplicity={u: [station_runway.item_uri(i)
                                  for i in evidence["future"]].count(u)
                              for u in uris},
        now=now_dt, db_path=db_path)
    batch["intended_map"] = resolved
    # Which picks were recognisable anchors (Discovery's cadence): carried
    # in memory only, so the ring entry can say WHY a track was programmed.
    # Older batches lack it; confirm_landed_batch uses .get().
    batch["anchor_paths"] = set(anchor_paths or ())

    base = dict(base, station_id=sid, queue_id=queue_id,
                intended_count=len(resolved), batch_id=batch["batch_id"])

    try:
        with station_lock.station_mutation():
            return _append_under_lock(
                station, queue_id, uris, batch, base, automatic=automatic,
                honour_switch=honour_switch, settings_fn=settings_fn,
                now_dt=now_dt, db_path=db_path, read=read,
                append_fn=append_fn, settle_fn=settle_fn)
    except station_lock.StationMutationBusy:
        # Nothing was written -- the write happens only inside the lock -- so
        # the claim is settled now rather than left for recovery, which could
        # not prove "never sent" once anything else joined the queue.
        station_store.resolve_batch(batch["batch_id"], station_store.ABANDONED,
                                    resolution="station_mutation_busy",
                                    now=now_dt, db_path=db_path)
        return _fail("station_mutation_busy", **base)


def _append_under_lock(station: dict[str, Any], queue_id: str, uris: list,
                       batch: dict[str, Any], base: dict[str, Any], *,
                       automatic: bool, honour_switch: bool, settings_fn,
                       now_dt: datetime, db_path, read: dict[str, Any],
                       append_fn, settle_fn=None) -> dict[str, Any]:
    """commit_append's half that runs INSIDE the station mutation lock."""
    import ma_queue
    sid = station["station_id"]
    # Revalidate INSIDE the lock. Everything above ran unserialized and
    # took seconds; this is the point where a Start Radio that won in the
    # meantime must stop a batch belonging to a station that no longer
    # exists.
    fresh = station_store.get_active_station(db_path=db_path)
    if fresh is None or fresh["station_id"] != sid \
            or (fresh.get("queue_id") or "") != queue_id:
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.ABANDONED,
                                    resolution="station_superseded",
                                    now=now_dt, db_path=db_path)
        return _fail("station_superseded", **base)
    if station_store.get_preparing_station(db_path=db_path) is not None:
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.ABANDONED,
                                    resolution="start_radio_in_flight",
                                    now=now_dt, db_path=db_path)
        return _fail("start_radio_in_flight", **base)
    still = station_store.get_pending_batch(sid, db_path=db_path)
    if still is None or still["batch_id"] != batch["batch_id"]:
        return _fail("claim_lost", **base)
    if automatic and honour_switch and not _keep_radio_playing(settings_fn):
        # The customer may switch automation off DURING selection, which
        # takes seconds. Eligibility said yes a moment ago; this is the
        # last word, and it belongs here because only here is it too late
        # to change our mind.
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.ABANDONED,
                                    resolution="keep_radio_playing_off",
                                    now=now_dt, db_path=db_path)
        return _fail("setting_off", **base)
    evidence = station_recovery.read_queue_evidence(queue_id, **read)
    if evidence is None:
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.ABANDONED,
                                    resolution="queue_unreadable",
                                    now=now_dt, db_path=db_path)
        return _fail("queue_unreadable_or_incomplete", **base)
    live_tail = station_recovery.ordered_queue(evidence)[-1]
    if live_tail.get("queue_item_id") != batch["pre_tail_queue_item_id"]:
        # The tail moved between the witness and the lock: the anchor the
        # proof depends on is no longer true, so refuse rather than write
        # something we could not later verify.
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.ABANDONED,
                                    resolution="tail_moved",
                                    now=now_dt, db_path=db_path)
        return _fail("tail_moved", **base)

    # THE ONE EXTERNAL WRITE. No retry on any outcome -- a failure and an
    # ambiguous response are the same thing here, and both are answered
    # by reading the queue, never by writing again.
    write_error = None
    try:
        (append_fn or ma_queue.append_tracks)(queue_id, uris)
    except Exception as exc:  # noqa: BLE001 - single write boundary
        write_error = str(exc)[:200]
        logger.warning("station append failed or was ambiguous: %s",
                       write_error)

    after = station_recovery.read_queue_evidence(queue_id, **read)
    if after is None:
        return {"ok": False, "reason": "reconciliation_unreadable",
                "appended_count": 0, "batch_state": station_recovery.UNPROVEN,
                "write_error": write_error, **base}
    verdict, reason = station_recovery.classify_append(batch, after)
    if (settle_fn is not None and write_error
            and verdict == station_recovery.NOT_LANDED):
        settle_fn()
        again = station_recovery.read_queue_evidence(queue_id, **read)
        if again is not None:
            after = again
            verdict, reason = station_recovery.classify_append(batch, after)
    if verdict == station_recovery.LANDED:
        appended = station_recovery.appended_items(batch, after)
        confirmed = confirm_landed_batch(station, batch, after,
                                         now=now_dt, db_path=db_path)
        fresh = station_store.get_station(sid, db_path=db_path) or {}
        if not confirmed:
            # The songs landed but this claim was settled by someone
            # else meanwhile. Success only if it was settled as landed;
            # anything else is reported, never papered over.
            state = (station_store.get_batch(batch["batch_id"],
                                             db_path=db_path) or {}).get(
                "state")
            if state != station_store.CONFIRMED:
                logger.error("station append landed but its claim %s is "
                             "%s: not confirmed here", batch["batch_id"][:8],
                             state)
                return {"ok": False, "reason": "confirmation_lost",
                        "appended_count": len(appended),
                        "batch_state": state, "reconciliation": reason,
                        "write_error": write_error, **base}
            confirmed = True
        return {"ok": True, "reason": "", "appended_count": len(appended),
                "batch_state": (station_store.CONFIRMED if confirmed
                                else station_store.PENDING),
                "reconciliation": reason,
                "witness_count": len(fresh.get("tail") or []),
                "witness_exact": not fresh.get("tail_is_partial"),
                "generated_count": len(fresh.get("generated") or []),
                "write_error": write_error, **base}
    if verdict == station_recovery.NOT_LANDED:
        station_store.resolve_batch(batch["batch_id"],
                                    station_store.NOT_LANDED,
                                    resolution=reason, now=now_dt,
                                    db_path=db_path)
        return {"ok": False, "reason": "append_did_not_land",
                "appended_count": 0, "batch_state": station_store.NOT_LANDED,
                "reconciliation": reason, "write_error": write_error, **base}
    # UNPROVEN: the batch stays pending on purpose. A later request or a
    # restart looks again; nothing appends in the meantime.
    return {"ok": False, "reason": "reconciliation_unproven",
            "appended_count": 0, "batch_state": station_store.PENDING,
            "reconciliation": reason, "write_error": write_error, **base}
