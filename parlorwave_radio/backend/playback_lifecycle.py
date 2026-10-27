#!/usr/bin/env python3
"""Session lifecycle for Playback Intelligence.

A session is a claim about the present: one active listening episode on one
player. This module ENFORCES that claim instead of assuming HA reports honestly:

- The reaper runs playback.cleanup() in-process on an interval (expired >
  max-age keyed on started_at, abandoned > idle window, TTL purge) -- a
  long-lived zombie session dies here, including one fed by a stale event
  source, whose last_event_at never goes stale.
- Supersede-on-start lives in playback.start_session (at most one active
  session per player); this module is the home for everything else lifecycle:
  by-player session resolution and playback adoption.

Binding principle: every ambiguity resolves toward ending stale state and
observing reality -- the backend never waits for HA to be honest.

Import direction: app.py -> playback_lifecycle -> playback (and musicdb for
catalog lookups). playback.py never imports this module.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import media_ref
import playback
import service_clock

logger = logging.getLogger("musicbrain")

# 0 disables the reaper (tests / rollback); the CLI `python -m playback cleanup`
# remains for manual passes either way.
REAPER_INTERVAL_MIN = float(os.environ.get("PLAYBACK_REAPER_INTERVAL_MIN", "5"))

# Playback adoption: a track_start arriving by-player with no active
# session creates an adopted session -- IF the track resolves unambiguously
# against the catalog. Non-library media (podcasts, Spotify, TTS chimes,
# doorbells) never adopts. PLAYBACK_ADOPT=0 disables (rollback).
ADOPT_ENABLED = os.environ.get("PLAYBACK_ADOPT", "1") == "1"

# --- What may start a listening session -------------------------------------
#
# Adoption used to require a local-library match, which gated on the wrong
# thing: it silently refused to observe legitimate streaming/unmatched music,
# while only incidentally excluding chimes and TTS. Catalog matching stays --
# for resolution, enrichment and scoring -- but it no longer decides whether
# playback is real.
#
# Replacing it needs a principled non-music guard, built on the strongest
# evidence the player actually reports rather than a duration guess.
#
# 1. Deny list on media_content_id. Deterministic and identity-based: audio
#    this process serves (breaks, news bulletins) and Home Assistant TTS have
#    stable URL shapes. This is the binding rule -- programming must NEVER
#    adopt a music session, whatever else it looks like. It is deliberately
#    the FIRST check, because a bulletin arrives carrying every mark of music:
#    content type 'music', a real ID3 artist, a five-minute duration.
# 2. media_content_type, when the player reports one. HA reports 'music' for
#    Music Assistant tracks regardless of provider, so this is what makes
#    adoption provider-agnostic.
# 3. Duration, ONLY when there is no type evidence at all. Deliberately last:
#    a 12-second interlude is real music, so duration must never override an
#    explicit media type.
# Structural, not substring. Each entry is a URI scheme or a rooted path
# segment, so a song legitimately CALLED "Announcement" -- or living in a
# folder named after one -- is never caught. The earlier bare "announcement"
# substring was exactly that bug.
NON_MUSIC_URI_SCHEMES = (
    "media_source://tts/",   # Home Assistant TTS media source
    "tts://",                # generic TTS scheme
)
#
# The programming segments come from media_ref so there is exactly ONE list of
# "audio Music Brain itself queued": breaks and news bulletins are both served
# by this process, and a bulletin is no more music than a break is. Adding a
# future programming type there covers adoption automatically instead of
# leaving a second list to drift out of sync -- which is precisely how news
# bulletins reached session_tracks as unresolved "songs".
NON_MUSIC_PATH_SEGMENTS = tuple(
    segment for _kind, segment in media_ref.PROGRAMMING_SEGMENTS
) + (
    "/api/tts_proxy/",       # Home Assistant TTS proxy
)
# Podcast/audiobook/video are deliberately absent: they are real playback but
# not music, and a listening session exists to learn music taste. Excluding
# them keeps podcast episodes out of play_events rather than polluting taste
# with them. Reversible in one line if that trade turns out wrong.
MUSIC_CONTENT_TYPES = frozenset({"music", "track"})
MIN_ADOPT_DURATION_S = float(os.environ.get("PLAYBACK_MIN_ADOPT_DURATION_S", "20"))
# Refusals that mean "the player told us nothing", as opposed to "the player
# told us this is not music". Only these may be overturned by a catalog match.
_EVIDENCE_ABSENT = frozenset({"no_media_type_or_duration",
                              "too_short_without_media_type"})


def is_adoptable_media(payload: dict[str, Any]) -> tuple[bool, str]:
    """May this track_start begin a listening session? Returns (ok, reason).

    The reason is recorded on refusal so a wrongly-rejected item is
    diagnosable from the response rather than vanishing silently.
    """
    content_id = str(payload.get("media_content_id") or "").strip().lower()
    for scheme in NON_MUSIC_URI_SCHEMES:
        if content_id.startswith(scheme):
            return False, f"non_music_scheme:{scheme.rstrip(':/')}"
    # Path segments are matched on the URI PATH only, so a query string or a
    # track title can never trip them.
    path_part = content_id.split("?", 1)[0].split("#", 1)[0]
    for segment in NON_MUSIC_PATH_SEGMENTS:
        if segment in path_part:
            return False, f"non_music_path:{segment.strip('/')}"

    content_type = str(payload.get("media_content_type") or "").strip().lower()
    if content_type:
        if content_type in MUSIC_CONTENT_TYPES:
            return True, "media_type"
        return False, f"non_music_type:{content_type}"

    duration = payload.get("duration")
    try:
        duration = None if duration is None else float(duration)
    except (TypeError, ValueError):
        duration = None
    if duration is None or duration <= 0:
        return False, "no_media_type_or_duration"
    if duration < MIN_ADOPT_DURATION_S:
        return False, "too_short_without_media_type"
    return True, "duration_fallback"


def _auto_adopt_enabled() -> bool:
    """Customer intent for OBSERVATION, read fresh so a settings change takes
    effect without a restart.

    Deliberately reads outputs.auto_adopt_playback and nothing else. dj.enabled
    is break permission and must never decide whether playback is observed, and
    outputs.enabled is the DJ's allow-list, not an observation allow-list --
    Music Brain keeps learning from what you play with the DJ switched off.
    Settings must never take ingestion down, so any failure falls back to the
    permissive default.
    """
    try:
        import settings
        return bool(settings.load()["dj"]["outputs"]["auto_adopt_playback"])
    except Exception:
        logger.warning("auto_adopt_playback unreadable; defaulting to enabled",
                       exc_info=True)
        return True


def get_active_for_queue(queue_id: str, db_path=None) -> dict[str, Any] | None:
    """The active session for a Music Assistant queue, or None. Thin re-export
    so callers have one lifecycle-facing surface for session resolution."""
    return playback.get_active_for_queue(queue_id, db_path=db_path)


def list_active_sessions(db_path=None) -> list[dict[str, Any]]:
    """Every active session, newest first. The canonical enumeration for
    single-station target resolution: an observer that follows "the active
    listening session" asks THIS, not its own reading of the sessions
    table, so there is exactly one interpretation of active."""
    playback.ensure_sessions_schema(db_path)
    c = playback._connect(db_path)
    try:
        rows = c.execute(
            f"SELECT {', '.join(playback._SESSION_COLS)} FROM sessions "
            "WHERE status = 'active' ORDER BY started_at DESC").fetchall()
    finally:
        c.close()
    return [playback._session_dict(r) for r in rows]


def retire_session_for_handoff(session_id: str, to_queue_id: str,
                               now: datetime | None = None,
                               db_path=None) -> bool:
    """Explicit station handoff: the user requested Start Radio on a
    DIFFERENT output while this session's player was paused/idle/gone.

    The in-flight track resolves through the ordinary inference ladder
    (source=inferred, never invented positive evidence -- the same honest
    mechanism supersede and pause_timeout use), the session ends, and a
    'station_handoff' event records where ownership went. The old player's
    MA transport is NEVER touched; a paused queue may stay paused forever."""
    now_dt = now if now is not None else service_clock.legacy_now()
    playback.ensure_sessions_schema(db_path)
    c = playback._connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(playback._SESSION_COLS)} FROM sessions "
            "WHERE session_id = ? AND status = 'active'",
            (session_id,)).fetchone()
        session = playback._session_dict(row) if row else None
    finally:
        c.close()
    if session is None:
        return False
    c = playback._connect(db_path)
    try:
        playback._resolve_current(
            c, session, ratio=playback._effective_ratio(session),
            explicit_skip=False, source="inferred",
            reason="station_handoff", now=now_dt)
        c.commit()
    finally:
        c.close()
    if not playback.end_session(session_id, status="ended", now=now_dt,
                                db_path=db_path):
        return False
    playback.append_event(session_id, "station_handoff",
                          {"to_queue_id": to_queue_id}, now=now_dt,
                          db_path=db_path, dedup=False)
    return True


def get_active_for_player(player: str, db_path=None) -> dict[str, Any] | None:
    """The active session for a player entity, or None. Session identity is a
    backend lookup, never HA state (a session id held in an HA helper can go
    stale; a backend lookup has nothing to go stale). Post-supersede there is at
    most one; newest wins defensively either way."""
    if not player:
        return None
    playback.ensure_sessions_schema(db_path)
    c = playback._connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(playback._SESSION_COLS)} FROM sessions "
            "WHERE player = ? AND status = 'active' "
            "ORDER BY started_at DESC LIMIT 1", (player,),
        ).fetchone()
    finally:
        c.close()
    return playback._session_dict(row) if row else None


def _catalog_match(path: str | None, media_title: str | None,
                   media_artist: str | None) -> str | None:
    """Resolve a player-reported track against the library catalog; the
    adoption gate. Returns the catalog path only on an unambiguous match
    (mirrors playback._match_track's fallback discipline)."""
    import musicdb
    c = musicdb.conn()
    try:
        if path:
            row = c.execute("SELECT path FROM tracks WHERE path = ? LIMIT 1",
                            (path,)).fetchone()
            return row[0] if row else None
        title = (media_title or "").strip()
        artist = (media_artist or "").strip()
        if not title or not artist:
            return None
        rows = c.execute(
            "SELECT path FROM tracks WHERE lower(artist) = lower(?) "
            "AND lower(title) = lower(?) LIMIT 2", (artist, title),
        ).fetchall()
    finally:
        c.close()
    return rows[0][0] if len(rows) == 1 else None


def handle_event_by_player(player: str, kind: str,
                           payload: dict[str, Any] | None = None,
                           now: datetime | None = None, adopt: bool | None = None,
                           db_path=None) -> dict[str, Any]:
    """Route a player event to that player's active session -- and when there
    is none, either adopt (catalog-matched track_start: someone is playing
    library music; ParlorWave joins as the same persistent host) or decline
    quietly. One Listening Session model: an adopted session differs only in
    origin metadata and queue_owned=False."""
    payload = payload or {}
    if not player:
        return {"ok": False, "error": "player required"}
    # Queue identity first: one MA queue is one session, however many player
    # entities report it. A group and each of its children all carry the
    # group's queue_id, so this is what stops one playback from becoming
    # several sessions. Falls back to player when the caller sends no
    # queue_id, so pre-queue_id forwarders behave exactly as before.
    queue_id = str(payload.get("queue_id") or "").strip()
    session = playback.get_active_for_queue(queue_id, db_path=db_path) if queue_id else None
    if session is None:
        session = get_active_for_player(player, db_path=db_path)
        if session is not None and queue_id:
            # First event to carry identity adopts the running session onto
            # its queue; a session already bound to another queue is untouched.
            playback.attach_queue_id(session["session_id"], queue_id, db_path=db_path)
            session = playback.get_session(session["session_id"], db_path=db_path) or session
    adopted = False
    if session is None:
        adopt_on = ADOPT_ENABLED if adopt is None else adopt
        if adopt_on and adopt is None:
            adopt_on = _auto_adopt_enabled()
        if not adopt_on:
            return {"ok": True, "handled": False, "reason": "adoption_disabled"}
        if kind != "track_start":
            # Only a track boundary can begin a session. A position report for
            # an unknown queue says nothing about what is playing.
            return {"ok": True, "handled": False, "reason": "no_active_session"}
        if not queue_id:
            # Identity first: without the owning queue we cannot tell one
            # output from another, and an anonymous session would be the exact
            # duplicate-per-entity failure queue identity exists to prevent.
            return {"ok": True, "handled": False, "reason": "no_queue_id"}
        # Catalog match is no longer the GATE -- it is enrichment. An unmatched
        # track is honestly unmatched: no path is invented for it, and
        # playback._match_track records it as such downstream.
        matched = _catalog_match(payload.get("path"), payload.get("media_title"),
                                 payload.get("media_artist"))
        adoptable, why = is_adoptable_media(payload)
        if not adoptable and matched and why in _EVIDENCE_ABSENT:
            # Demoted, not deleted: a positive catalog match proves this is a
            # music track even when the player reported no type and no usable
            # duration. It can only rescue MISSING evidence -- never override
            # an explicit deny or a stated non-music type, so a DJ break
            # that somehow matched a catalog row is still refused.
            adoptable, why = True, "catalog_match"
        if not adoptable:
            logger.info("declined adoption on %s (%s): %s - %s", player, why,
                        payload.get("media_artist"), payload.get("media_title"))
            return {"ok": True, "handled": False, "reason": why}
        session = playback.start_session(
            request="", mode="default_adaptive", player=player,
            origin="adopted", queue_owned=False, queue_id=queue_id,
            now=now, db_path=db_path)
        adopted = True
        logger.info("adopted session %s on %s queue=%s (%s) %s - %s [%s]",
                    session["session_id"], player, queue_id, why,
                    payload.get("media_artist"), payload.get("media_title"),
                    "catalog" if matched else "unmatched")
    result = playback.handle_event(session["session_id"], kind, payload,
                                   now=now, db_path=db_path)
    result["session_id"] = session["session_id"]
    if queue_id:
        result["queue_id"] = queue_id
    if adopted:
        result["adopted"] = True
    return result


def reaper_pass(now: datetime | None = None, db_path=None) -> dict[str, int]:
    """One maintenance sweep: expire over-age actives, abandon idle actives,
    purge past-TTL terminals. Thin, testable seam over playback.cleanup()."""
    return playback.cleanup(now=now, db_path=db_path)


def _station_recovery_pass() -> dict[str, Any]:
    """Reconcile a stranded `preparing` station. READ-ONLY against Music
    Assistant by construction: station_recovery has no path that can write a
    queue, because an external write whose outcome is unknown must never be
    repeated on a guess. SERIALIZED against live station writes: a proof
    read in the middle of an append would misjudge it."""
    import station_recovery
    result = station_recovery.recover_pass_serialized()
    if result.get("checked"):
        logger.info("station recovery: %s", result)
    return result


def _station_live_pass() -> dict[str, Any]:
    """Keep a Station Programmer (Mode 2) station's runway. The reaper is
    its clock on a queue the observer is not following: one runway check
    per interval, and a refill only when the station's own music has run
    low. A no-op (one local read) when no such station is on air."""
    import station_live
    return station_live.maintain(trigger="reaper")


async def run_reaper(interval_min: float | None = None) -> None:
    """Long-running reaper loop for the FastAPI lifespan. Runs one pass
    immediately (deploy is self-healing: boot retires every zombie), then
    sweeps on the interval. Errors are logged, never fatal -- a failed sweep
    must not take the service down with it."""
    interval = REAPER_INTERVAL_MIN if interval_min is None else interval_min
    if interval <= 0:
        return
    while True:
        try:
            stats = reaper_pass()
            if any(stats.values()):
                logger.info("lifecycle reaper: %s", stats)
        except Exception:
            logger.exception("lifecycle reaper pass failed")
        # A station preparation stranded by a crash is reconciled here: boot
        # runs one pass immediately, so a restart resolves it before anything
        # else looks at the queue. Deliberately OFF the event loop -- it makes
        # bounded Music Assistant READS (never a write), and blocking the loop
        # on network I/O would stall every request. Fenced separately from the
        # DB sweep so neither failure can hide the other.
        try:
            await asyncio.to_thread(_station_recovery_pass)
        except Exception:
            logger.exception("station recovery pass failed")
        # After recovery on purpose: a stranded start or refill is settled
        # by reading the queue first, and only then may the runway grow.
        try:
            await asyncio.to_thread(_station_live_pass)
        except Exception:
            logger.exception("station programmer pass failed")
        await asyncio.sleep(interval * 60.0)
