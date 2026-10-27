#!/usr/bin/env python3
"""📻 Start Radio From This Song -- a first-class Music Brain intent.

The listener presses one button (or, later, asks for it by voice) and the
FUTURE of the queue becomes a station seeded from the
song that is playing right now:

    current song keeps playing, untouched
      -> two more tracks by the seed artist (the loved behavior carried
         over from the old station logic; fewer when the library has fewer)
      -> a broader station built from the seed song + artist through the
         existing grounded station machinery (LLM picks WEIGHTS for library
         artists only -- it can never invent a song; deterministic
         genre-based fallback when the LLM is unavailable)

HARD INVARIANT: pressing the button never interrupts the current song.
There is no pause/stop/play/seek/next anywhere in this path. The single
queue mutation is MA QueueOption 'replace_next', which replaces everything
AFTER the current item and leaves the current item and its position alone
(option enum verified against MA).

DJ coordination: an ON-AIR break refuses politely (the 'current
song' is the DJ); a QUEUED break is revoked first (its queue item is
superseded by the new future -- the row must go terminal so its audio
answers 410 and the scheduler plans a fresh, correctly-grounded break);
planned/rendering/ready breaks are left alone -- the scheduler's existing
stale-context validation re-renders them against the new queue.

Every selected track must resolve to a real MA library URI before anything
is written; unresolvable picks are dropped, and below a minimum the queue
is left completely unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import media_ref

logger = logging.getLogger("musicbrain")


class _AckSkipped(Exception):
    """The acknowledgement liner is deliberately not produced (e.g. the DJ
    is off). Never an error: the station change already succeeded."""


# The rolling-runway direction: a listenable finite future, not another
# 500-track blob. Replenishment tops it up (station_refill).
DEFAULT_RUNWAY = 40
SEED_ARTIST_LEAD = 2          # the "two more from this artist" behavior
MIN_RESOLVED = 5              # below this, refuse and leave the queue alone
RESOLVE_SEARCH_LIMIT = 8


def _norm(s: str | None) -> str:
    return (s or "").strip().casefold()


def _seed_artist_picks(seed_artist: str, seed_title: str,
                       tracks: list[dict[str, Any]],
                       scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Up to SEED_ARTIST_LEAD distinct catalog tracks by the seed artist,
    best effective_score first, never the currently-playing title."""
    from llm_station_builder import allowed, effective_score
    own = [t for t in tracks
           if _norm(t.get("artist")) == _norm(seed_artist)
           and _norm(t.get("title")) != _norm(seed_title)
           and allowed(t, scores, seed_artist)]
    own.sort(key=lambda t: effective_score(t, scores), reverse=True)
    picks: list[dict[str, Any]] = []
    seen_titles = set()
    for t in own:
        key = _norm(t.get("title"))
        if key in seen_titles:
            continue
        picks.append(t)
        seen_titles.add(key)
        if len(picks) >= SEED_ARTIST_LEAD:
            break
    return picks


def _genre_fallback(seed_artist: str, seed_title: str,
                    tracks: list[dict[str, Any]],
                    count: int) -> list[dict[str, Any]]:
    """Deterministic broader station when the LLM is unavailable: artists
    sharing the seed track's genre (else the seed artist's genres), ranked
    through the same select_tracks scoring/spacing machinery.

    Returns (picks, artist_weights, request_text) -- the weights and the
    request are the frozen station profile, identical in shape to the LLM
    path's."""
    from llm_station_builder import select_tracks
    from musicdb import get_scores_for_artists
    genres = {_norm(t.get("genre")) for t in tracks
              if _norm(t.get("artist")) == _norm(seed_artist)
              and _norm(t.get("genre"))}
    if genres:
        artists = {t.get("artist") for t in tracks
                   if _norm(t.get("genre")) in genres and t.get("artist")}
    else:
        artists = {seed_artist}
    weights = {a: 5 for a in artists}
    weights[seed_artist] = 8
    pool = [t for t in tracks if t.get("artist") in artists]
    scores = get_scores_for_artists(list(artists))
    request = f"music like {seed_artist} - {seed_title}"
    # The deterministic path produces a weights dict too, so BOTH creation
    # paths yield a freezable profile and a refill never needs the LLM.
    return (select_tracks(weights, pool, scores, request, count),
            weights, request)


def _broader_station(seed_artist: str, seed_title: str,
                     tracks: list[dict[str, Any]], count: int,
                     builder_fn: Callable[[str, int], dict[str, Any]] | None,
                     ) -> tuple[list[dict[str, Any]], str, dict, str]:
    """The post-lead station: the existing grounded /station machinery (the
    LLM only weights artists already in the library; select_tracks then
    chooses REAL catalog tracks), with the deterministic genre fallback.

    Returns (picks, source, artist_weights, request_text). The last two are
    the frozen semantic profile persisted with the station; the LLM still runs
    exactly ONCE, here, exactly as it does today."""
    request = f"music similar to {seed_artist} - {seed_title}"
    if builder_fn is None:
        from llm_station_builder import build_station
        builder_fn = build_station
    try:
        result = builder_fn(request, count)
        picked = result.get("tracks") or []
        if picked:
            # The LLM's artist WEIGHTS are the station's semantic identity and
            # were previously thrown away here, which is why a station could
            # never be replenished without asking the LLM again. Returned now
            # so start_from_current can freeze them; selection policy and
            # taste deliberately stay live.
            return picked, "station", dict(result.get("artists") or {}), request
    except Exception:
        logger.warning("station builder failed; using genre fallback",
                       exc_info=True)
    picks, weights, fallback_request = _genre_fallback(
        seed_artist, seed_title, tracks, count)
    return picks, "genre", weights, fallback_request


def _resolve_to_uris(picks: list[dict[str, Any]],
                     search_fn: Callable[[str, int], list[dict[str, Any]]],
                     limit: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve catalog picks to REAL MA library URIs by title search +
    exact artist/title match. Unresolvable picks are dropped -- nothing
    invented ever reaches the queue."""
    uris: list[str] = []
    resolved: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    for t in picks:
        if len(uris) >= limit:
            break
        title, artist = t.get("title"), t.get("artist")
        if not title or not artist:
            continue
        try:
            hits = search_fn(title, RESOLVE_SEARCH_LIMIT)
        except Exception:
            continue
        for h in hits:
            if (_norm(h.get("name")) == _norm(title)
                    and any(_norm(a) == _norm(artist)
                            for a in h.get("artists") or [])
                    and h.get("uri") and h["uri"] not in seen_uris):
                uris.append(h["uri"])
                seen_uris.add(h["uri"])
                resolved.append(t)
                break
    return uris, resolved


def _station_activity(queue_id: str, players: list[dict[str, Any]],
                      state_fn: Callable[[str], Any]) -> str:
    """'playing' | 'inactive' | 'unknown' for the OLD station's output.

    Only genuinely audible playback blocks an explicit handoff: a player
    that is paused, idle, unavailable, or gone from the MA inventory is
    not a station anyone is listening to. 'unknown' (unreadable state)
    refuses conservatively -- never hand off what might be audible."""
    import ma_queue
    p = next((x for x in players if isinstance(x, dict)
              and x.get("player_id") == queue_id), None)
    if p is None or not p.get("available", False):
        return "inactive"                    # left the inventory / dead
    try:
        qs = state_fn(queue_id)
    except ma_queue.MAQueueError as exc:
        if "not found" in str(exc):
            return "inactive"                # queue vanished entirely
        return "unknown"
    except Exception:
        return "unknown"
    state = (qs or {}).get("state")
    if state == "playing":
        return "playing"
    if state in ("paused", "idle", None, ""):
        return "inactive"
    return "unknown"


def _commit_output_intent(queue_id: str) -> dict[str, Any]:
    """Record a SUCCESSFUL explicit Start Radio as station-output intent:
    X becomes the selected/default output AND gains DJ speech permission.

    Additive on permission: other enabled outputs are preserved, and the
    global dj.enabled flag is never touched -- an explicit station start
    grants where-the-DJ-may-speak, never whether-the-DJ-speaks-at-all.
    Called only after the queue mutation committed; its own failure is
    reported honestly but never unwinds the station (music already won)."""
    import settings
    try:
        outs = dict(settings.load()["dj"]["outputs"])
        enabled = list(outs.get("enabled") or [])
        if queue_id not in enabled:
            enabled.append(queue_id)
        settings.save_section("dj", {"outputs": {
            "default_request_output": queue_id,
            "enabled": enabled}})
        return {"committed": True,
                "default_request_output": queue_id,
                "speech_enabled": True}
    except Exception:
        logger.warning("output-intent commit failed (the station change "
                       "itself stands)", exc_info=True)
        return {"committed": False}


def _pre_write_witness(queue_id: str, state_fn=None, future_fn=None
                       ) -> dict[str, Any]:
    """The compact evidence that lets a crashed preparation be judged later.

    Music Assistant returns nothing useful from a write, so the ONLY way to
    later prove "the replacement landed" versus "it never happened" is to know
    what the queue looked like immediately before. A count plus an ordered
    hash of the future's music URIs costs one column each instead of a queue
    dump.

    Best effort by contract: an unreadable queue yields the schema's
    not-captured sentinel (-1). That weakens only the NOT_LANDED half of the
    later proof -- LANDED is still provable from the intended URIs alone --
    and it must never block a station change that works today."""
    blank = {"pre_current_queue_item_id": "", "pre_future_music_count": -1,
             "pre_future_music_hash": ""}
    try:
        import station_recovery
        evidence = station_recovery.read_queue_evidence(
            queue_id, state_fn=state_fn, future_fn=future_fn)
        if evidence is None:
            return blank
        music = station_recovery.future_music_uris(evidence["future"])
        import station_store
        return {"pre_current_queue_item_id": evidence["current_queue_item_id"],
                "pre_future_music_count": len(music),
                "pre_future_music_hash": station_store.uri_list_hash(music)}
    except Exception:
        logger.warning("pre-write witness unavailable (station change "
                       "proceeds)", exc_info=True)
        return blank


def _prepare_station(queue_id: str, uris: list[str], seed_artist: str,
                     seed_title: str, seed_media_key: str,
                     artist_weights: dict, profile_request: str,
                     broad_source: str, witness: dict, now=None,
                     db_path=None) -> dict[str, Any] | None:
    """Write the station's intent BEFORE the queue mutation, or return None.

    Returning None means "no durable identity for this station" -- which is
    exactly today's behavior, and today's behavior works. Persistence is
    additive here: it must never be able to refuse a station change that the
    live-proven path would have made.

    A preparation stranded by an earlier crash is resolved first, and if it
    still cannot be proven it is SUPERSEDED: an explicit Start Radio is the
    definitive supersession signal in this architecture, and a preparation
    that never won is not something to protect. Without this a single stranded
    row would block every future station from being persisted, because the
    schema permits only one preparing station at a time."""
    import station_recovery
    import station_store
    try:
        stranded = station_store.get_preparing_station(db_path=db_path)
        if stranded is not None:
            station_recovery.recover_pass(now=now, db_path=db_path)
            stranded = station_store.get_preparing_station(db_path=db_path)
        if stranded is not None:
            station_store.deactivate_station(
                stranded["station_id"], "superseded_by_new_request",
                now=now, db_path=db_path)
        return station_store.create_preparing_station(
            request_text=profile_request, artist_weights=artist_weights,
            seed_artist=seed_artist, seed_title=seed_title,
            seed_media_key=seed_media_key, profile_source=broad_source,
            queue_id=queue_id, intended_uris=uris, now=now, db_path=db_path,
            **witness)
    except Exception:
        logger.warning("station preparation not persisted (the station "
                       "change itself proceeds)", exc_info=True)
        return None


def _confirm_station(station: dict[str, Any], queue_id: str, now=None,
                     db_path=None, state_fn=None, future_fn=None) -> str:
    """Prove the replacement from the queue, then activate -- or leave it.

    Returns the verdict. UNPROVEN deliberately leaves the row `preparing` and
    the OLD station `active`: Music Assistant has already been written, so
    re-issuing the write would be the one genuinely destructive response to
    uncertainty. The lifecycle recovery pass looks again later."""
    import station_recovery
    try:
        evidence = station_recovery.read_queue_evidence(
            queue_id, state_fn=state_fn, future_fn=future_fn)
        if evidence is None:
            return station_recovery.UNPROVEN
        return station_recovery.resolve_preparing(station, evidence, now=now,
                                                  db_path=db_path)
    except Exception:
        logger.warning("station activation could not be proven (queue "
                       "already changed; recovery will retry)", exc_info=True)
        return "unproven"


def _acquire_ownership(station_id: str, queue_id: str, *, now=None,
                       db_path=None, state_fn=None, autoplay_read_fn=None,
                       autoplay_write_fn=None) -> dict[str, Any] | None:
    """Fenced: ownership acquisition can never turn a committed station
    change into a failure. Returns the acquisition record or None.

    The pre-ownership flag comes from the same queue snapshot the proof
    already reads (player_queues/all), so an install that does not report it
    is recorded as unread rather than guessed."""
    try:
        import ma_queue
        import queue_ownership
        observed, err = None, ""
        try:
            flag = ((state_fn or ma_queue.queue_state)(queue_id) or {}
                    ).get("autoplay_enabled")
            observed = flag if isinstance(flag, bool) else None
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            err = str(exc)[:200]
        return queue_ownership.acquire(
            queue_id=queue_id, station_id=station_id, observed=observed,
            observed_error=err, now=now, db_path=db_path,
            read_fn=autoplay_read_fn, write_fn=autoplay_write_fn)
    except Exception:
        logger.warning("queue ownership acquisition failed (station change "
                       "stands; retried at the next refill evaluation)",
                       exc_info=True)
        return None


def start_from_current(queue_id: str | None = None,
                       count: int = DEFAULT_RUNWAY,
                       current_fn=None, search_fn=None, replace_fn=None,
                       builder_fn=None, library_fn=None, ack_deps=None,
                       inventory_fn=None, sessions_fn=None, state_fn=None,
                       future_fn=None, now=None, db_path=None,
                       autoplay_read_fn=None, autoplay_write_fn=None
                       ) -> dict[str, Any]:
    """The whole intent. Returns {ok, ...} and NEVER touches playback on
    any failure path -- an honest refusal leaves the queue exactly as it
    was.

    An EXPLICIT queue_id is "Start Radio on output X": deliberate user
    intent that X (any valid MA player id -- Chromecast UUID, sendspin
    ma_*, hass entity) is the ParlorWave station. Explicit targets are
    validated against the live MA inventory (never guessed, never
    silently replaced with a fallback), refused with
    active_station_elsewhere while a different queue owns the active
    listening session (single station: nothing is ever teleported or
    doubled), and -- only after the station commit succeeds -- recorded
    as dj.outputs.default_request_output plus speech permission for X.
    Callers that omit queue_id keep today's behavior exactly."""
    import ma_queue
    import station_lock

    # A Station Programmer (Mode 2) station owns programming wherever it
    # plays. Start Radio would supersede it -- silently ending its refill on
    # another speaker -- so it is refused until that station is stopped.
    try:
        import station_store
        if station_store.is_programmer(station_store.get_active_station(
                db_path=db_path)) or station_store.is_programmer(
                station_store.get_preparing_station(db_path=db_path)):
            return {"ok": False, "reason": "station_programmer_active"}
    except Exception:  # noqa: BLE001 - persistence never blocks Start Radio
        # Same doctrine as _prepare_station: an unreadable store must not
        # refuse a station change the live-proven path would make.
        logger.warning("station state unreadable before Start Radio "
                       "(proceeding)", exc_info=True)

    explicit = bool(queue_id)
    handoff_sessions: list[dict[str, Any]] = []
    # 0. Which queue? Start Radio seeds from what is PLAYING, so the target
    # is resolved by the shared control policy -- the active station's queue
    # first, then the one enabled output actually playing. It is never
    # `outputs.enabled[0]`: that list says where DJ behaviour is allowed,
    # not where the DJ is, and anchoring here refused a live reseed with
    # `nothing_playing` because a silent speaker happened to sort first.
    if not queue_id:
        import output_target
        picked = output_target.resolve(None, require_playing=True,
                                       state_fn=state_fn)
        if not picked.get("ok"):
            return {"ok": False, "reason": picked["reason"],
                    **({"candidates": picked["candidates"]}
                       if picked.get("candidates") else {})}
        queue_id = picked["queue_id"]
    if not queue_id:
        return {"ok": False, "reason": "no_queue"}

    if explicit:
        # 0b. The target must exist in the live MA inventory. An unknown id
        # is refused outright -- no name-guessing, no named-player fallback.
        try:
            players = (inventory_fn or ma_queue.list_players)()
            known = {p.get("player_id") for p in players
                     if isinstance(p, dict)}
        except Exception:
            return {"ok": False, "reason": "inventory_unreadable"}
        if queue_id not in known:
            return {"ok": False, "reason": "unknown_output"}
        # 0c. SINGLE STATION, state-aware: an audibly PLAYING station on
        # another queue is never teleported and never doubled. But when the
        # old station is paused/idle/gone (MA surfaces Pause, not Stop, so
        # sessions linger in pause grace), the explicit request IS the
        # user's decision -- those sessions become handoff candidates,
        # retired only after the new station actually commits.
        try:
            import playback_lifecycle
            actives = (sessions_fn
                       or playback_lifecycle.list_active_sessions)()
        except Exception:
            return {"ok": False, "reason": "session_state_unreadable"}
        elsewhere = [s for s in actives
                     if (s.get("queue_id") or "")
                     and s["queue_id"] != queue_id]
        for s in elsewhere:
            activity = _station_activity(s["queue_id"], players,
                                         state_fn or ma_queue.queue_state)
            if activity == "playing":
                return {"ok": False, "reason": "active_station_elsewhere",
                        "active_queue_id": s["queue_id"]}
            if activity == "unknown":
                return {"ok": False, "reason": "station_state_unreadable",
                        "active_queue_id": s["queue_id"]}
            handoff_sessions.append(s)
        if handoff_sessions:
            # A handoff must move to a station that is actually audible:
            # the requested output has to be PLAYING real music right now,
            # or the old (merely paused) session stays untouched.
            try:
                xstate = (state_fn or ma_queue.queue_state)(queue_id)
            except Exception:
                return {"ok": False, "reason": "queue_unreadable"}
            if (xstate or {}).get("state") != "playing":
                return {"ok": False, "reason": "target_not_playing"}

    # 1. Canonical current item from MA.
    try:
        current = (current_fn or ma_queue.current_item_detail)(queue_id)
    except Exception:
        return {"ok": False, "reason": "queue_unreadable"}
    if not current or not current.get("title"):
        return {"ok": False, "reason": "nothing_playing"}
    # Programming is never a seed. A break was already refused here; a news
    # bulletin carries a real ID3 artist (its source name), so without this it would
    # have resolved and built an actual station out of a news bulletin. The
    # existing 'jarvis_on_air' reason is preserved exactly for breaks.
    prog_kind, _prog_token = media_ref.programming_ref(current.get("uri"))
    if prog_kind:
        return {"ok": False,
                "reason": "jarvis_on_air" if prog_kind == "break"
                          else f"{prog_kind}_on_air"}
    seed_artist = (current.get("artists") or [None])[0]
    seed_title = current.get("title")
    if not seed_artist:
        return {"ok": False, "reason": "artist_unresolved"}

    # 2. Selection from the CATALOG (real tracks only).
    try:
        import library_repo
        tracks = (library_fn or library_repo.get_library)()["tracks"]
    except Exception:
        return {"ok": False, "reason": "library_unavailable"}
    # An install with no local catalog cannot start a station from a song,
    # however the request arrived -- the UI gate, a voice command, or a
    # direct POST. Refused HERE, from the catalog this call actually read,
    # so the gate and the selection can never disagree; and refused with
    # the sentence a person is shown, not a code a person has to look up.
    # MIN_RESOLVED is the same floor step 3 enforces after resolution: a
    # catalog that cannot offer that many candidates can never clear it.
    if len(tracks) < MIN_RESOLVED:
        import catalog_status
        return {"ok": False, "reason": catalog_status.REASON,
                "message": catalog_status.MESSAGE,
                "track_count": len(tracks)}
    from musicdb import get_scores_for_artists
    seed_scores = get_scores_for_artists([seed_artist])
    lead = _seed_artist_picks(seed_artist, seed_title, tracks, seed_scores)
    broader, broad_source, artist_weights, profile_request = _broader_station(
        seed_artist, seed_title, tracks, count, builder_fn)

    taken = {(_norm(seed_artist), _norm(seed_title))}
    ordered: list[dict[str, Any]] = []
    for t in lead + broader:
        key = (_norm(t.get("artist")), _norm(t.get("title")))
        if key in taken:
            continue
        taken.add(key)
        ordered.append(t)
        if len(ordered) >= count:
            break

    # 3. Resolve every pick to a REAL MA URI; drop the unresolvable.
    uris, resolved = _resolve_to_uris(
        ordered, search_fn or ma_queue.search_library_tracks, count)
    lead_keys = {(_norm(t.get("artist")), _norm(t.get("title")))
                 for t in lead}
    lead_names = [f"{t.get('artist')} - {t.get('title')}" for t in resolved
                  if (_norm(t.get("artist")), _norm(t.get("title")))
                  in lead_keys]
    if len(uris) < MIN_RESOLVED:
        return {"ok": False, "reason": "could_not_resolve_station",
                "resolved": len(uris)}

    # 4. DJ coordination BEFORE the mutation.
    import dj_engine
    revoked_breaks: list[str] = []
    revoked_queued = None          # frozen response field (additive-only API)
    try:
        c = dj_engine._connect(db_path)
        try:
            rows = c.execute(
                "SELECT token, state FROM dj_breaks WHERE queue_id = ? AND "
                f"state IN ({','.join('?' for _ in dj_engine.NON_TERMINAL_STATES)})",
                (queue_id, *dj_engine.NON_TERMINAL_STATES)).fetchall()
        finally:
            c.close()
        if any(state == "airing" for _t, state in rows):
            return {"ok": False, "reason": "jarvis_on_air"}
        # Every other in-flight break is superseded by the new future: a
        # queued item's audio goes 410, and planned/rendering/ready copy was
        # grounded against a runway that no longer exists, so it is revoked
        # rather than replanned -- deterministic instead of relying on
        # stale-context validation. This also frees the one-in-flight slot
        # for the acknowledgement liner. (Music Brain state only: MA may
        # already have BUFFERED a queued item, which no revocation can
        # recall -- deliberately not fought with transport actions.)
        for token, state in rows:
            if dj_engine.revoke(token, now, db_path, reason="station_change"):
                revoked_breaks.append(token)
                if state == "queued":
                    revoked_queued = token
    except Exception:
        logger.warning("break coordination failed; refusing station change",
                       exc_info=True)
        return {"ok": False, "reason": "dj_coordination_failed"}

    # 5. ONE queue mutation: replace the FUTURE only -- serialized against
    # every other station-owned MA write, and bracketed by durable identity.
    #
    # The lock is GLOBAL, not per-queue, because Start Radio reaches across
    # outputs: a station-owned writer still in flight for the OLD station's
    # queue must not land after the listener has moved the station here. A
    # busy lock means no station change at all -- never a write anyway.
    #
    # Inside it, and in this order: capture the pre-write witness, commit the
    # `preparing` row (the old station stays ACTIVE and truthful), write to
    # MA, read the queue back, and only then swap the two rows in one
    # transaction. No SQLite transaction is ever open across the network call.
    station = None
    station_verdict = None
    ownership = None
    try:
        lock_cm = station_lock.station_mutation()
    except Exception:                       # pragma: no cover - defensive
        lock_cm = None
    if lock_cm is None:
        return {"ok": False, "reason": "station_mutation_unavailable"}
    try:
        with lock_cm:
            # Re-checked under the lock: a Station Programmer start that won
            # the race since the check above must not be superseded here.
            try:
                import station_store
                if station_store.is_programmer(
                        station_store.get_active_station(db_path=db_path)) \
                        or station_store.is_programmer(
                            station_store.get_preparing_station(
                                db_path=db_path)):
                    return {"ok": False,
                            "reason": "station_programmer_active"}
            except Exception:  # noqa: BLE001 - persistence never blocks
                logger.warning("station state unreadable under the lock "
                               "(Start Radio proceeds)", exc_info=True)
            witness = _pre_write_witness(queue_id, state_fn, future_fn)
            station = _prepare_station(
                queue_id, uris, seed_artist, seed_title,
                current.get("uri") or "", artist_weights, profile_request,
                broad_source, witness, now=now, db_path=db_path)
            try:
                (replace_fn or ma_queue.replace_future)(queue_id, uris)
            except Exception:
                logger.warning("station replace_future failed (queue "
                               "unchanged)", exc_info=True)
                if station is not None:
                    # The queue never changed, so the preparation definitively
                    # did not land. The OLD station is untouched and keeps
                    # playing -- exactly today's failure semantics.
                    import station_store
                    station_store.fail_prepared_station(
                        station["station_id"], now=now, db_path=db_path)
                return {"ok": False, "reason": "queue_update_failed"}
            if station is not None:
                station_verdict = _confirm_station(
                    station, queue_id, now=now, db_path=db_path,
                    state_fn=state_fn, future_fn=future_fn)
                import station_recovery
                if station_verdict == station_recovery.LANDED:
                    # Queue ownership: the station is ACTIVE on this
                    # queue, so ParlorWave is its music programmer from here on
                    # and Music Assistant's own Autoplay is switched off for
                    # THIS queue -- still inside the lock, so nothing
                    # station-owned can write between activation and the
                    # flag change. Never a station failure: an unverified
                    # flag is retried at the first refill evaluation.
                    ownership = _acquire_ownership(
                        station["station_id"], queue_id, now=now,
                        db_path=db_path, state_fn=state_fn,
                        autoplay_read_fn=autoplay_read_fn,
                        autoplay_write_fn=autoplay_write_fn)
    except station_lock.StationMutationBusy:
        logger.warning("station mutation lock busy; no station change made")
        return {"ok": False, "reason": "station_mutation_busy"}

    # 5a. Explicit handoff: the old station's session is retired ONLY now,
    # after the new station actually committed, so a failed rebuild leaves
    # the old ownership fully intact. The old player's MA transport is
    # never touched -- a paused queue may stay paused forever. A retire
    # failure at this point is reported, never repaired with transport.
    handoff = None
    if handoff_sessions:
        import playback_lifecycle
        retired = []
        for s in handoff_sessions:
            try:
                if playback_lifecycle.retire_session_for_handoff(
                        s["session_id"], queue_id, now=now, db_path=db_path):
                    retired.append(s["session_id"])
            except Exception:
                logger.warning("station handoff retirement failed for %s",
                               s["session_id"], exc_info=True)
        handoff = {"from_queue_id": handoff_sessions[0]["queue_id"],
                   "retired_sessions": retired}

    # 5b. Explicit target = deliberate ownership intent, recorded ONLY
    # after the queue commit so a failed station can never leave a
    # misleading selection or speech permission behind. Runs before the
    # liner so the ack's own permission check already sees the intent.
    output_intent = None
    if explicit:
        output_intent = _commit_output_intent(queue_id)

    # 6. Best-effort acknowledgement liner, created ONLY now: the station is
    # committed, so the liner can never claim a success that did not happen.
    # Rendering runs in the background (the button must not wait on LLM/TTS)
    # and every liner failure is silent -- the station change stands, the
    # first new track plays next, and ordinary scheduling resumes.
    # The liner is DJ speech: when the DJ is off for this queue the station
    # still changes, silently. Fail toward silence, exactly like plan_break.
    # With DJ SPEECH off the liner is skipped the same way, under its own
    # reason: the DJ's lifecycle is running, only its voice is not.
    ack_token = None
    ack_skipped = None
    try:
        if not dj_engine.dj_enabled_for_queue(queue_id):
            raise _AckSkipped("dj_disabled")
        if not dj_engine.dj_speech_enabled():
            raise _AckSkipped(dj_engine.SPEECH_DISABLED)
        ack_token = dj_engine.create_request_ack(
            queue_id,
            {"artist": seed_artist,
             "song": f"{seed_artist} - {seed_title}",
             "first_new_track": lead_names[0] if lead_names else None,
             "queue_item_id": current.get("queue_item_id")},
            now=now, db_path=db_path)
        dj_engine.start_request_ack_render(ack_token, deps=ack_deps, now=now,
                                           db_path=db_path)
    except _AckSkipped as skipped:
        ack_token = None
        ack_skipped = str(skipped)
    except Exception:
        logger.warning("station-request acknowledgement skipped "
                       "(station change stands)", exc_info=True)
        ack_token = None

    return {"ok": True,
            "queue_id": queue_id,
            # Additive only -- every pre-existing field keeps its meaning.
            # station_id is None when durable identity could not be written,
            # which is precisely today's behavior and never a station failure.
            "station_id": (station or {}).get("station_id"),
            "station_state": station_verdict,
            # Queue ownership: what happened to MA Autoplay on this queue.
            "ownership": ownership,
            "output_intent": output_intent,
            "handoff": handoff,
            "seed": {"artist": seed_artist, "title": seed_title},
            "seed_lead_tracks": lead_names,
            "runway": len(uris),
            "broader_source": broad_source,
            "ack_token": ack_token,
            # Why there is no liner when that was deliberate (dj_disabled /
            # dj_speech_disabled); None when a liner was made or failed.
            "ack_skipped": ack_skipped,
            "revoked_breaks": revoked_breaks,
            "revoked_queued_break": revoked_queued,
            "note": (None if len(lead_names) >= SEED_ARTIST_LEAD else
                     f"only {len(lead_names)} additional seed-artist "
                     "track(s) available; broadened early")}
