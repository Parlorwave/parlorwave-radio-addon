#!/usr/bin/env python3

import asyncio
import datetime as _dt
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

import re as _re

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt
import logging_setup
import playback
import playback_arcs
import playback_lifecycle
# The queue-hygiene engine. Imported statically HERE, and only here,
# so it is part of the backend's import closure (release/export.py ships
# exactly what the entrypoints reach) without any exporter special case.
# The two routes below are its only callers; nothing schedules it.
import queue_hygiene
import settings
from musicdb import init_db, feedback
from llm_station_builder import build_station, interpret_mode
from lidarr_client import get_lidarr_status
from collector import (
    collection_summary,
    collection_completion,
    collection_gaps,
    collection_artist,
    missing_albums_from_ledger,
)

# Music Brain's single logging configuration point. Done at import of the
# ASGI app because `uvicorn app:app` is how production starts: uvicorn
# configures only its own loggers and leaves root at WARNING, so without this
# every logger.info() in the package is discarded. See logging_setup.
logging_setup.configure()

logger = logging.getLogger("musicbrain")

# Authentication is the product default. Every route except GET / (health)
# and the token-scoped finite-audio routes needs X-Api-Key. The key comes
# from config (MUSICBRAIN_API_KEY, else the persisted key file, else one is
# generated at startup); an EMPTY key fails closed. MUSICBRAIN_AUTH_DISABLED=1
# is the explicit, loud development/test escape -- never the default.
import config as _config

_CFG = _config.current()
API_KEY = _CFG.api_key
AUTH_DISABLED = _CFG.auth_disabled
#: this installation's stable identity (config.ensure_instance_id), read at
#: startup. The Home Assistant integration keys its config entry on it, so
#: it must not change when the host, address or API key does.
INSTANCE_ID = _CFG.instance_id
#: when THIS process started. /dj/status carries it, so the integration
#: sees a restart it did not catch failing -- the add-on restarts in a few
#: seconds when a token is pasted into it, usually between two polls --
#: and does not go on describing the process that came before.
SERVICE_STARTED = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if AUTH_DISABLED:
        return
    if not API_KEY:
        raise HTTPException(status_code=503,
                            detail="API key not initialised; see /health/problems")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-Api-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global API_KEY, INSTANCE_ID
    # Startup self-check: refuse to run on a host that is not configured,
    # with one line per missing setting; feature gaps are warnings. As the
    # Home Assistant add-on the customer's own settings arrive later through
    # the add-on's options, so those are feature problems, not fatal ones.
    cfg = _config.current()
    fatal = _config.required_problems(cfg)
    if fatal:
        for p in fatal:
            logger.error("config: %s -> %s", p["summary"], p["action"])
        raise _config.ConfigError(fatal)
    # This installation's identity, before anything can ask for it.
    INSTANCE_ID, id_source = _config.ensure_instance_id(cfg)
    if id_source == "generated":
        logger.info("installation id created at %s", cfg.instance_id_path)
    if AUTH_DISABLED:
        logger.warning("MUSICBRAIN_AUTH_DISABLED=1 -- API authentication is "
                       "OFF; every endpoint is open. Development only.")
    else:
        API_KEY, source = _config.ensure_api_key(cfg)
        if source == "generated":
            logger.warning("API key generated and stored at %s -- give it "
                           "to the Home Assistant integration", cfg.api_key_path)
        else:
            logger.info("API key source: %s", source)
    for p in _config.feature_problems(cfg):
        logger.warning("config: %s -> %s", p["summary"], p["action"])
    logger.info("config: %s", _config.describe(cfg))
    init_db()
    # Restart recovery: verify/create sessions.db (archives on version mismatch).
    # Active sessions resume as-is from disk; the next HA event re-synchronizes.
    playback.ensure_sessions_schema()
    # A DJ render is work inside ONE process. A break the previous process
    # left 'rendering' can never finish and would hold every break until the
    # stale-break TTL, so it is revoked here -- before the observer below
    # can drive the planner. Renders are pre-handoff: nothing reached MA.
    try:
        import dj_engine
        dj_engine.recover_orphaned_renders()
    except Exception:
        logger.exception("orphaned DJ render recovery failed")
    # In-process lifecycle reaper (expire > max-age, abandon > idle,
    # purge > TTL). First pass runs at boot -- deploy retires zombies itself.
    reaper = asyncio.create_task(playback_lifecycle.run_reaper())
    # The Music Assistant observer: the product's playback authority
    # (config default authoritative/auto; MA_OBSERVER_MODE overrides).
    import ma_observer
    mode = ma_observer.start_from_env()
    if mode != ma_observer.MODE_AUTH:
        logger.warning("MA observer is %s (product default: authoritative); "
                       "playback is not followed and no break can be "
                       "scheduled from Music Assistant", mode)
    try:
        yield
    finally:
        reaper.cancel()
        ma_observer.stop()


def _docs_kwargs(cfg: "_config.Config") -> dict:
    """The interactive API docs are a development aid, never a product
    surface: an unauthenticated /openapi.json enumerates every route.
    MUSICBRAIN_DEBUG_DOCS=1 enables them explicitly."""
    if cfg.debug_docs:
        return {}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(title="ParlorWave Radio service", lifespan=lifespan,
              **_docs_kwargs(_CFG))


class StationRequest(BaseModel):
    prompt: str
    count: int = Field(30, ge=1, le=100)


class FeedbackRequest(BaseModel):
    action: Literal["thumbs_up", "thumbs_down", "skip", "played"]
    artist: str | None = None
    title: str | None = None
    path: str | None = None


class MissingAlbumsRequest(BaseModel):
    prompt: str
    count: int = Field(50, ge=1, le=100)
    format: str = "lidarr"


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ParlorWave Radio",
        "auth": "disabled" if AUTH_DISABLED else "required",
    }


@app.get("/instance", dependencies=[Depends(require_api_key)])
def instance():
    """This installation's stable identity.

    The Home Assistant integration keys its config entry on `instance_id`,
    so that a backend whose hostname, address or API key changes is still
    recognised as the SAME installation instead of becoming a second one.
    Authenticated, because nothing about this service should be readable by
    an unauthenticated caller that has not already been let in; the id
    itself is an opaque random name, not a secret and not a fingerprint."""
    out = {"ok": True, "instance_id": INSTANCE_ID,
           "service": "ParlorWave Radio",
           "deployment": "addon" if _config.current().addon else "standalone"}
    # The release this service was built as (the add-on image sets it from
    # its build version). Absent rather than guessed when unknown: the
    # integration compares it with its own and stays silent without it.
    version = (os.environ.get("MUSICBRAIN_VERSION") or "").strip()
    if version:
        out["version"] = version
    return out


@app.get("/health/problems", dependencies=[Depends(require_api_key)])
def health_problems():
    """Structured, actionable configuration and connectivity facts: what is
    fatal, what is broken, what is merely unconfigured. Read-only; the one
    live check is a bounded Music Assistant probe."""
    import health
    return health.report()


@app.post("/station", dependencies=[Depends(require_api_key)])
def station(req: StationRequest):
    import catalog_status
    try:
        result = build_station(req.prompt, req.count)
    except catalog_status.CatalogUnavailable as exc:
        # A supported state, not a fault: this install follows Music
        # Assistant instead of programming music itself. Said in one
        # finished sentence, and never logged as an exception.
        return {"ok": False, "reason": catalog_status.REASON,
                "error": str(exc), "stdout": ""}
    except Exception as exc:
        logger.exception("station build failed")
        return {
            "ok": False,
            "error": "station build failed; see server log",
            "stdout": ""
        }

    station_name = result.get("station_name", req.prompt)
    artists = {"raw": result.get("artists_text", "")}
    tracks = []

    for track in result.get("tracks", []):
        tracks.append({
            "artist": track.get("artist", ""),
            "title": track.get("title", ""),
            "album": track.get("album", ""),
            "display": f"{track.get('artist', '')} - {track.get('title', '')}"
        })

    return {
        "ok": True,
        "station": station_name,
        "prompt": req.prompt,
        "count": len(tracks[:req.count]),
        "artists": artists,
        "tracks": tracks[:req.count]
    }


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback_endpoint(req: FeedbackRequest):
    try:
        changed = feedback(
            path=req.path,
            artist=req.artist,
            title=req.title,
            action=req.action
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Best-effort session mirror: an explicit skip/thumb during an active
    # session becomes a session event (and an explicit skip resolves the current
    # track). Never alters this endpoint's response or raises.
    try:
        playback.mirror_feedback(req.action, path=req.path, artist=req.artist,
                                 title=req.title)
    except Exception:
        logger.exception("session feedback mirror failed")
    return {
        "ok": True,
        "action": req.action,
        "updated_rows": changed
    }


@app.get("/lidarr/status", dependencies=[Depends(require_api_key)])
def lidarr_status():
    return get_lidarr_status()


@app.post("/missing_albums", dependencies=[Depends(require_api_key)])
def missing_albums_endpoint(req: MissingAlbumsRequest):
    # Served from the recommendation ledger (last generation run) -- a pure read.
    # No MusicBrainz, no Ollama on the request path; generation is background-only
    # (python -m collector recommend). Shape frozen; freshness fields additive.
    return missing_albums_from_ledger(req.prompt, req.count, req.format)


# Collector Intelligence (read-only; served from the materialized collection.db).
@app.get("/collection/summary", dependencies=[Depends(require_api_key)])
def collection_summary_endpoint():
    return collection_summary()


@app.get("/collection/completion", dependencies=[Depends(require_api_key)])
def collection_completion_endpoint(limit: int = Query(50, ge=0, le=500)):
    return collection_completion(limit=limit)


@app.get("/collection/gaps", dependencies=[Depends(require_api_key)])
def collection_gaps_endpoint(limit: int = Query(20, ge=0, le=500)):
    return collection_gaps(limit=limit)


@app.get("/collection/artist/{name:path}", dependencies=[Depends(require_api_key)])
def collection_artist_endpoint(name: str):
    return collection_artist(name)


# --- Playback Intelligence (session state engine) ----------------------------
# Sessions live in sessions.db (ephemeral, TTL'd); permanent play history goes to
# play_events. No MusicBrainz, no recommendation ledger, no Lidarr on any
# playback path. The temporary intended queue reuses build_station() -- the same
# function behind /station -- per the approved design.

# Optional LLM mode mapping at session start: opt-in via env because it
# adds one bounded Ollama call to session start. Off, the chain is
# explicit -> keyword -> default_adaptive; on, the LLM slots in after explicit.
PLAYBACK_MODE_LLM = os.environ.get("PLAYBACK_MODE_LLM", "0") == "1"


class PlaybackSessionRequest(BaseModel):
    # mode "" = auto-select from the utterance (explicit -> optional LLM ->
    # keyword -> default_adaptive). An explicit valid mode always wins.
    request: str = ""
    mode: str = ""
    player: str = ""
    count: int = Field(30, ge=1, le=100)
    context: dict | None = None
    queue: bool = True  # False: start the session without a station-built queue


class PlaybackEventRequest(BaseModel):
    kind: Literal["track_start", "position", "skip", "idle", "stop",
                  "paused", "resumed"]
    # Music Assistant queue identity (the owning player/group's player_id, which
    # HA exposes as the media_player `active_queue` attribute). Optional: when
    # absent the backend falls back to player-keyed resolution exactly as before.
    queue_id: str | None = None
    path: str | None = None
    media_title: str | None = None
    media_artist: str | None = None
    # Media identity/type evidence for the non-music adoption guard. Optional:
    # absent means the guard falls back to duration, which is exactly the
    # pre-existing behaviour for any forwarder that has not been updated.
    media_content_id: str | None = None
    media_content_type: str | None = None
    # Album as the PLAYER reported it. Recorded verbatim for items with no
    # catalog row; never synthesised, so an unresolved item cannot acquire
    # library metadata it never had.
    media_album_name: str | None = None
    position: float | None = None
    duration: float | None = None
    previous_position: float | None = None
    previous_duration: float | None = None


@app.post("/playback/listener_play", dependencies=[Depends(require_api_key)])
def playback_listener_play(body: dict):
    """A PERSON asked this queue's speaker to play: the integration saw a
    Home Assistant play service call that carried a user. A NOTE for the
    end-of-queue tracker (queue_end), so a restart seconds after a queue
    ended counts as the listener's new pass rather than MA re-opening a
    Flow by itself. It changes nothing: no transport, no queue, no setting."""
    import time
    import queue_end
    queue_id = str((body or {}).get("queue_id") or "").strip()
    if not queue_id:
        raise HTTPException(status_code=422, detail="queue_id is required")
    queue_end.note_listener_play(queue_id, time.time())
    return {"ok": True}


@app.post("/playback/session", dependencies=[Depends(require_api_key)])
def playback_session_start(req: PlaybackSessionRequest):
    # Mode selection: explicit -> optional bounded LLM (session start
    # ONLY; /next, /refill, /event never construct an llm_fn) -> keyword ->
    # default_adaptive. Invalid/unavailable LLM output falls through silently.
    llm_fn = None
    if PLAYBACK_MODE_LLM and not req.mode and req.request:
        llm_fn = lambda u: interpret_mode(u, sorted(playback_arcs.VALID_MODES))
    mode, mode_source = playback_arcs.resolve_mode(req.mode, req.request, llm_fn=llm_fn)
    session = playback.start_session(request=req.request, mode=mode,
                                     player=req.player, context=req.context)
    queue: list[dict] = []
    queue_error = None
    queue_reason = None
    if req.queue:
        import catalog_status
        try:
            result = build_station(req.request, req.count)
            queue = result.get("tracks", [])
            playback.record_queue(session["session_id"], queue, source="station")
        except catalog_status.CatalogUnavailable as exc:
            # No local catalog: a request cannot be filled here, and that
            # is a supported state. The session still starts (Principle 6)
            # and the refusal travels as the finished sentence a caller can
            # show, with the stable reason beside it for a gate to read.
            queue_error = str(exc)
            queue_reason = catalog_status.REASON
        except Exception:
            # Principle 6: the session still starts; the queue can be refilled.
            logger.exception("playback session queue build failed")
            queue_error = "station build failed; see server log"
    session = playback.get_session(session["session_id"])
    out = {"ok": True, "session": session, "mode_source": mode_source, "queue": [
        {"path": t.get("path"), "artist": t.get("artist"), "title": t.get("title"),
         "album": t.get("album")} for t in queue
    ]}
    if queue_error:
        out["queue_error"] = queue_error
    if queue_reason:
        out["queue_reason"] = queue_reason
    return out


@app.get("/playback/session/{session_id}", dependencies=[Depends(require_api_key)])
def playback_session_get(session_id: str):
    session = playback.get_session(session_id)
    if session is None:
        return {"ok": False, "error": "session not found"}
    return {"ok": True, "session": session}


@app.post("/playback/session/{session_id}/event", dependencies=[Depends(require_api_key)])
def playback_session_event(session_id: str, req: PlaybackEventRequest):
    payload = {k: v for k, v in req.model_dump().items() if k != "kind" and v is not None}
    return playback.handle_event(session_id, req.kind, payload)


class SongRequestBody(BaseModel):
    # Pandora-aware request surface. Parsed artist/title intent comes from the
    # existing upstream request parser -- this endpoint only does source
    # resolution/execution underneath it (no new NLP here). An empty
    # queue_id (the panel's Request Line, alpha.4) means "the speaker the DJ
    # is on now", resolved here by output_target, never guessed.
    queue_id: str = ""
    artist: str = ""
    title: str = ""


@app.post("/request/song", dependencies=[Depends(require_api_key)])
def request_song(req: SongRequestBody):
    """Studio / HA-conversation song request surface (Pandora-aware). Resolves
    LOCAL-first and, on a confident hit, inserts the local track as ONE
    user-intent item next in the queue -- inside a Pandora session that is a
    one-off local intervention that leaves Pandora's chosen order and refill
    untouched. Ambiguous / artist-only / miss return a STRUCTURED result the
    conversational layer speaks; a local miss carries the provider truth
    (current Pandora cannot exact-play an arbitrary requested track). Every
    request surface calls this same resolver."""
    import pandora_runtime
    if req.queue_id:
        return pandora_runtime.handle_local_request(
            req.queue_id, req.artist or None, req.title or None)
    # The panel's Request Line (alpha.4): ONE song, next, on the speaker the
    # DJ is on now -- pandora_runtime.request_next owns every rule
    # (never while paused, never past MA's buffer, re-read before and
    # read back after the one insert, one pending request per speaker).
    import output_target
    tgt = output_target.resolve(None, require_playing=True)
    if not tgt.get("ok"):
        return {"action": "none", "reason": tgt.get("reason") or "no_target"}
    return pandora_runtime.request_next(tgt["queue_id"], req.artist or None,
                                        req.title or None)


@app.post("/playback/session/{session_id}/feedback", dependencies=[Depends(require_api_key)])
def playback_session_feedback(session_id: str, req: FeedbackRequest):
    try:
        changed = feedback(path=req.path, artist=req.artist, title=req.title,
                           action=req.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    mirrored = playback.mirror_feedback(req.action, path=req.path, artist=req.artist,
                                        title=req.title, session_id=session_id)
    return {"ok": True, "action": req.action, "updated_rows": changed,
            "session": mirrored[0] if mirrored else None}


@app.post("/playback/session/{session_id}/end", dependencies=[Depends(require_api_key)])
def playback_session_end(session_id: str):
    if not playback.end_session(session_id):
        return {"ok": False, "error": "session not found or not active"}
    return {"ok": True, "session": playback.get_session(session_id)}


# --- By-player identity -------------------------------------------------------
# Session identity is a backend lookup, never HA state: HA automations address
# the PLAYER; the backend resolves (or adopts) the session. With these routes a
# single generic automation per event type serves any number of players, and
# the input_text session-id helper is deleted.

@app.get("/playback/session/by-player/{entity_id}",
         dependencies=[Depends(require_api_key)])
def playback_session_by_player(entity_id: str):
    session = playback_lifecycle.get_active_for_player(entity_id)
    if session is None:
        return {"ok": False, "error": "no active session for player"}
    return {"ok": True, "session": playback.get_session(session["session_id"])}


@app.post("/playback/session/by-player/{entity_id}/event",
          dependencies=[Depends(require_api_key)])
def playback_event_by_player(entity_id: str, req: PlaybackEventRequest):
    payload = {k: v for k, v in req.model_dump().items() if k != "kind" and v is not None}
    return playback_lifecycle.handle_event_by_player(entity_id, req.kind, payload)


@app.post("/playback/session/by-player/{entity_id}/feedback",
          dependencies=[Depends(require_api_key)])
def playback_feedback_by_player(entity_id: str, req: FeedbackRequest):
    try:
        changed = feedback(path=req.path, artist=req.artist, title=req.title,
                           action=req.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    session = playback_lifecycle.get_active_for_player(entity_id)
    mirrored = []
    if session is not None:
        mirrored = playback.mirror_feedback(req.action, path=req.path,
                                            artist=req.artist, title=req.title,
                                            session_id=session["session_id"])
    return {"ok": True, "action": req.action, "updated_rows": changed,
            "session": mirrored[0] if mirrored else None}


# --- Settings (product configuration) ----------------------------------------
# One persistent, versioned document behind section-scoped routes (dj,
# playback, bulletins, station); engine-level concerns get their own route
# when they acquire real customer-facing settings, and the scoping means
# adding one changes nothing here. There is deliberately no session-lifecycle section: a
# playback attachment ends when playback ends, never on a clock.
#
# Identity is the Music Assistant player_id, never a Home Assistant entity_id.
# Nothing consumes these values yet -- wiring lands with the adoption/broadcast
# steps, so this pass changes no playback behaviour.

def _get_settings_section(section: str):
    return {"ok": True, "section": section,
            "settings": settings.get_section(section)}


def _put_settings_section(section: str, patch: dict):
    """Partial update: send only the keys you intend to change. The merged
    result is validated as a whole, so a partial update can never persist an
    invalid document -- and the write cannot leave `section`, nor change a
    voice value one of the /voices routes owns (settings.VOICE_OWNED_PATHS:
    sending one back unchanged is fine)."""
    try:
        return {"ok": True, "section": section,
                "settings": settings.save_section(section, patch,
                                                  generic=True)}
    except settings.SettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/settings/dj", dependencies=[Depends(require_api_key)])
def get_dj_settings():
    return _get_settings_section("dj")


@app.put("/settings/dj", dependencies=[Depends(require_api_key)])
def put_dj_settings(patch: dict):
    """Partial patch of the dj section. One check beyond section validation:
    a voice (persona.voice.refs.chatterbox_voice) must be one the TTS
    service actually offers right now -- 422 for an unknown voice, 503 when
    the inventory cannot be read (an unverifiable voice would silently fail
    every future break). Personality is validated by the schema itself, as
    is the deprecated `persona.chattiness` compatibility key -- which is
    still accepted and still persisted, and which normal production no
    longer reads. Nothing here touches playback."""
    _validate_voice_patch(patch)
    return _put_settings_section("dj", patch)


def _validate_voice_patch(patch: dict) -> None:
    import tts_voices
    refs = patch
    for key in ("persona", "voice", "refs"):
        # a level that is not an object is the section validation's to
        # refuse (422), not this check's to trip over
        refs = refs.get(key) if isinstance(refs, dict) else None
    if not isinstance(refs, dict) or "chatterbox_voice" not in refs:
        return
    try:
        refs["chatterbox_voice"] = tts_voices.validate_choice(
            refs["chatterbox_voice"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except tts_voices.VoiceCatalogUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=f"cannot verify voice: TTS voice list unavailable ({exc})")


@app.get("/settings/services", dependencies=[Depends(require_api_key)])
def get_services_settings():
    """The two optional services. Credentials are NEVER in this answer:
    each key comes back null beside an `api_key_set` boolean, which is the
    only thing a screen needs.

    The provider vocabulary travels WITH the settings rather than being
    written into the panel. Which protocols the writer can speak is a
    backend fact -- the integration consumes this contract and names no
    inference host of its own, which is a standing invariant and pinned as
    one."""
    import optional_services
    return {**_get_settings_section("services"),
            "writer_providers": [
                {"id": "ollama", "label": "Ollama"},
                {"id": "openai_compatible", "label": "OpenAI-compatible"},
            ],
            "status": optional_services.status()}


@app.put("/settings/services", dependencies=[Depends(require_api_key)])
def put_services_settings(patch: dict):
    """Partial patch of the services section.

    A null api_key means LEAVE THE STORED ONE -- the read above returns
    null for every key, so a screen that reads the section and sends it
    back would otherwise blank a credential it was never shown. An empty
    string is the deliberate, and only, way to clear one."""
    return _put_settings_section("services", patch)


@app.get("/services/status", dependencies=[Depends(require_api_key)])
def services_status():
    """What each optional service IS, without contacting either. Safe to
    poll: it makes no network call and holds no credential."""
    import optional_services
    return {"ok": True, "services": optional_services.status()}


@app.post("/services/{kind}/test", dependencies=[Depends(require_api_key)])
def services_test(kind: str, override: dict | None = None):
    """Test Connection: one attempt, one short deadline, no retry.

    READ ONLY in every sense that matters -- it renders no speech, writes
    no break, touches no queue and no player, and cannot change what the
    station is about to do. A failure is a reason code and a sanitised
    detail; a remote service's own error text is scrubbed of anything
    credential-shaped before it travels, because a service may echo our
    Authorization header back inside it."""
    import optional_services
    if kind not in (optional_services.VOICE_ENGINE, optional_services.WRITER):
        raise HTTPException(status_code=404, detail="no such service")
    # An address on the screen is tested as it is on the screen. The body is
    # used for this call and stored nowhere -- Save is still the only thing
    # that writes, and a key that travels here is used once for one header.
    return optional_services.probe(kind, override or None)


@app.post("/services/writer/find", dependencies=[Depends(require_api_key)])
def services_find_writer(body: dict | None = None):
    """Look for an Ollama-shaped service at a short list of named places.

    Bounded by construction: the address on the screen, then three
    well-known names for the Home Assistant machine, each with a 1.5 s
    deadline. It is NOT a scanner -- nothing here enumerates a network or
    an address range -- and it answers with everything it tried so the
    screen can say where it looked. Read-only, like every other probe."""
    import optional_services
    body = body or {}
    return optional_services.find_writer(
        body.get("base_url") or "",
        provider=str(body.get("provider") or "ollama"))


@app.get("/settings/station", dependencies=[Depends(require_api_key)])
def get_station_settings():
    return _get_settings_section("station")


@app.put("/settings/station", dependencies=[Depends(require_api_key)])
def put_station_settings(patch: dict):
    """Station programming feel. Writing it is ONLY writing it: the value
    is read by the NEXT automatic replenishment of whatever station is
    active. No queue rebuild, no removal, no transport, no station
    identity change -- ever."""
    return _put_settings_section("station", patch)


@app.get("/tts/health", dependencies=[Depends(require_api_key)])
def tts_health():
    """READ ONLY: is the Kokoro speech sidecar reachable, and is routing
    allowed to use it? Operations and rollout need exactly this much --
    whether the kill switch is on, and whether the service answers. The
    Chatterbox path keeps the surfaces it already had; nothing here
    changes which voice speaks."""
    import tts_router
    return {"ok": True, "kokoro": tts_router.health()}


@app.get("/studio/catalog", dependencies=[Depends(require_api_key)])
def studio_catalog():
    """READ ONLY: the option lists a Studio UI may offer -- personalities,
    station feels and the TTS voices available right now. The voice list
    degrades honestly (ok=False + reason) when the service cannot be
    asked; nothing is written or moved.

    `chattiness` is still in the payload and is DEPRECATED: persona and
    airtime are decoupled, no shipped UI offers the list
    any more, and picking any of it changes nothing. The key stays so an
    older integration pointed at this backend still gets a well-formed
    catalog instead of an entity with no options."""
    import dj_persona
    import station_feel
    import tts_voices
    return {"ok": True,
            "personas": dj_persona.catalog(),
            # DEPRECATED -- see the docstring. Every entry now describes
            # the same behaviour because every entry now HAS the same
            # behaviour.
            "chattiness": [
                {"id": None, "name": "Standard",
                 "description": "The standard amount of talk: a callback "
                                "or aside when it earns its place."},
                {"id": "low", "name": "Low (no longer used)",
                 "description": "Retired. The DJ talks the standard "
                                "amount whatever is selected here."},
                {"id": "normal", "name": "Normal",
                 "description": "The standard amount of talk: a callback "
                                "or aside when it earns its place."},
                {"id": "high", "name": "High (no longer used)",
                 "description": "Retired. The DJ talks the standard "
                                "amount whatever is selected here."}],
            "feels": station_feel.catalog(),
            "voices": tts_voices.catalog()}


# --- Queue-native DJ finite audio ---------------------------------------------
# PUBLIC like GET /: Music Assistant fetches this URL itself and cannot send
# X-Api-Key. The 128-bit single-purpose token IS the credential -- it addresses
# exactly one rendered break file (path re-derived from the token, never from
# stored input). Once the break's lifecycle turns terminal the token answers
# the identified retired-program placeholder instead, never that break's
# stale speech and never a transport-fatal status -- see program_audio for
# why, and for the X-Parlorwave-Program header that says which it is. MA may
# probe/HEAD/Range-read the file any number of times before it airs; fetches
# are never treated as airtime.

def _serve_finite_audio(path, request: Request) -> Response:
    """The finite-audio HTTP contract shared by DJ breaks and news
    bulletins: GET/HEAD, single byte Range with 206/416, no-store, and
    repeatable reads (MA probes before it plays, browsers seek). Sized from
    stat and read by slice, so a HEAD or a Range never pulls a whole
    bulletin into memory.

    Every answer says which of two things it is. A retired token is
    deliberately still a 200 carrying a short ordinary track -- that is the
    transport contract and a history replay depends on it -- but it is
    no longer anonymous: the placeholder carries its own ID3 identity and
    this header names it, so a few seconds of silence can never be mistaken
    for a real break that rendered badly."""
    import program_audio
    total = path.stat().st_size
    # Content-Length is per-RESPONSE, never shared: a 416 carries an empty
    # body, so advertising the artifact's full size there is an h11 framing
    # error (an ASGI exception).
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store",
               program_audio.PROGRAM_HEADER:
                   program_audio.PROGRAM_RETIRED
                   if program_audio.is_tombstone(path)
                   else program_audio.PROGRAM_LIVE}
    rng = (request.headers.get("range") or "").strip()
    m = _re.match(r"^bytes=(\d*)-(\d*)$", rng) if rng else None
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), total - 1) if m.group(2) else total - 1
        else:  # suffix range: the last N bytes
            length = min(int(m.group(2)), total)
            start, end = total - length, total - 1
        if start >= total or start > end:
            return Response(status_code=416,
                            headers={**headers,
                                     "Content-Range": f"bytes */{total}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        headers["Content-Length"] = str(end - start + 1)
        if request.method == "HEAD":
            return Response(status_code=206, media_type="audio/mpeg",
                            headers=headers)
        with path.open("rb") as f:
            f.seek(start)
            chunk = f.read(end - start + 1)
        return Response(content=chunk, status_code=206,
                        media_type="audio/mpeg", headers=headers)
    headers["Content-Length"] = str(total)
    if request.method == "HEAD":
        return Response(status_code=200, media_type="audio/mpeg",
                        headers=headers)
    return Response(content=path.read_bytes(), media_type="audio/mpeg",
                    headers=headers)


@app.head("/broadcast/audio/{token_file}")
@app.get("/broadcast/audio/{token_file}")
def broadcast_audio(token_file: str, request: Request):
    import dj_engine
    token = token_file[:-4] if token_file.endswith(".mp3") else token_file
    status, path = dj_engine.audio_available(token)
    if status != 200:
        raise HTTPException(status_code=status)
    resp = _serve_finite_audio(path, request)
    if request.method == "GET" and _whole_file(resp, request):
        # DELIVERY evidence: the whole
        # file left for the player. Not proof of air -- MA prefetches -- and
        # recorded only after the body was sent.
        from starlette.background import BackgroundTask
        resp.background = BackgroundTask(dj_engine.note_served, token,
                                         int(path.stat().st_size))
    return resp


def _whole_file(resp: Response, request: Request) -> bool:
    """Did this response carry the whole file? A plain 200, or a 206 whose
    range is the entire file (Music Assistant asks 'bytes=0-')."""
    if resp.status_code == 200:
        return not request.headers.get("range")
    if resp.status_code == 206:
        m = _re.match(r"^bytes (\d+)-(\d+)/(\d+)$",
                      resp.headers.get("content-range") or "")
        return bool(m) and int(m.group(1)) == 0 and \
            int(m.group(2)) == int(m.group(3)) - 1
    return False


@app.get("/broadcast/waveform/current", dependencies=[Depends(require_api_key)])
def broadcast_current_waveform(queue_id: str | None = Query(None)):
    """READ ONLY: normalized waveform for the currently-playing audio track
    or program element from Music Assistant's audio analysis cache."""
    import waveform
    return waveform.get_current_waveform(queue_id)


@app.head("/news/audio/{token_file}")
@app.get("/news/audio/{token_file}")
def news_audio(token_file: str, request: Request):
    """The cached bulletin's own audio. PUBLIC by design, exactly like the
    break endpoint: Music Assistant fetches it itself and cannot send
    X-Api-Key, so the 128-bit token IS the credential -- and it resolves
    only to the file that token owns, never to 'whatever is current'."""
    import news_bulletins
    token = token_file[:-4] if token_file.endswith(".mp3") else token_file
    status, path = news_bulletins.audio_available(token)
    if status != 200:
        raise HTTPException(status_code=status)
    return _serve_finite_audio(path, request)


# --- voice library -----------------------------------------------------------

class VoiceSelectRequest(BaseModel):
    voice_id: str


class VoiceRenameRequest(BaseModel):
    name: str


class VoiceNoteRequest(BaseModel):
    note: str


class VoiceTuningRequest(BaseModel):
    speed: float
    pause: float


class VoicePreviewRequest(BaseModel):
    voice_id: str
    speed: float | None = None
    pause: float | None = None
    # a short line the owner typed; bounded by voice_library.preview_text
    text: str | None = None


class VoiceAnnouncementDefaultRequest(BaseModel):
    voice_id: str = ""


class VoiceScopeRequest(BaseModel):
    scope: str


@app.get("/voices/library", dependencies=[Depends(require_api_key)])
def voices_library():
    """READ ONLY: the built-in and custom voice shelves, live-verified
    availability, and the current selection."""
    import voice_library
    return voice_library.library()


@app.post("/voices/select", dependencies=[Depends(require_api_key)])
def voices_select(req: VoiceSelectRequest):
    """Select the DJ voice. Writes dj.persona.voice ONLY -- personality,
    chattiness, feel, programming and transport are untouched.

    The owner's explicit choice first proves the voice speaks
    (voice_library.speak_check): a voice that cannot is refused in words
    and the DJ stays as it was, so the Booth never shows a DJ that breaks
    cannot be spoken with."""
    import voice_library
    try:
        voice_library.speak_check(req.voice_id)
        return {"ok": True, **voice_library.select(req.voice_id)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


class VoiceShiftsRequest(BaseModel):
    enabled: bool | None = None
    morning: str | None = None
    afternoon: str | None = None
    night: str | None = None


@app.post("/voices/shifts", dependencies=[Depends(require_api_key)])
def voices_shifts(req: VoiceShiftsRequest):
    """Automatic DJ shifts: on/off, and which DJ-allowed voice (a logical
    id, "" to unassign) hosts Morning, Afternoon and Night. Writes
    dj.shifts -- and, when switching on or changing the current shift's
    voice, settles the current shift through the ordinary selection. A
    voice newly assigned first proves it speaks (voice_library.speak_check,
    one short phrase, nothing played); one that cannot is refused in words
    and nothing is saved. Never reaches Music Assistant."""
    import functools
    import voice_library
    assignments = {s: getattr(req, s) for s in ("morning", "afternoon", "night")
                   if getattr(req, s) is not None}
    prove = functools.partial(voice_library.speak_check,
                              refused=voice_library.REFUSED_FOR_SHIFT)
    try:
        return {"ok": True, **voice_library.set_shifts(req.enabled, assignments,
                                                       prove=prove)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.post("/voices/switch-dj", dependencies=[Depends(require_api_key)])
def voices_switch_dj():
    """"Switch DJ": select the next voice allowed to be the DJ, through
    the same select() -- writes dj.persona.voice ONLY. The next voice first
    proves it speaks (voice_library.speak_check, one short phrase, nothing
    played); one that cannot is refused in words and the DJ stays. Nothing
    reaches Music Assistant; the next DJ break rendered uses the new voice.
    The answer names who is now selected."""
    import voice_library
    try:
        return {"ok": True, **voice_library.switch_dj(
            prove=voice_library.speak_check)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.post("/voices/{voice_id}/rename", dependencies=[Depends(require_api_key)])
@app.post("/voices/custom/{voice_id}/rename",
          dependencies=[Depends(require_api_key)])     # pre-rename-built-ins path
def voices_rename(voice_id: str, req: VoiceRenameRequest):
    """A display name for any shelved voice; an empty name gives a
    built-in its curated name back. 422 for an unknown voice, a name that
    is too long, or one another voice already uses."""
    import voice_library
    try:
        return {"ok": True, **voice_library.rename(voice_id, req.name)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.post("/voices/{voice_id}/note", dependencies=[Depends(require_api_key)])
def voices_note(voice_id: str, req: VoiceNoteRequest):
    """The owner's reminder for a voice ("calm, slower reader"). Presentation
    only: it never reaches a prompt, an asset or a provider. 422 for a
    voice the shelf does not know or a note that is too long."""
    import voice_library
    try:
        return {"ok": True, **voice_library.set_note(voice_id, req.note)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


def _voice_failure(exc) -> HTTPException:
    """One mapping from a voice refusal to an HTTP answer, so no caller
    invents its own. The body is ALWAYS {"code", "message"}: a stable code
    a UI can branch on, and a sentence a person can read. A provider's own
    exception text, its URL and the environment variable that configures it
    never appear -- those went to the log where they belong."""
    import voice_library
    code = getattr(exc, "code", "voice_refused")
    status = 422
    if isinstance(exc, voice_library.VoiceEngineUnavailable):
        status = 503
    elif isinstance(exc, voice_library.VoiceEngineFailed):
        status = 502
    elif isinstance(exc, voice_library.VoiceSlotsFull):
        status = 409
    elif isinstance(exc, voice_library.VoiceEnrollmentUnavailable):
        status = 403
    return HTTPException(status_code=status,
                         detail={"code": code, "message": str(exc)})


def _voice_unexpected(exc, what: str) -> HTTPException:
    """Anything the voice modules did not classify. The customer still gets
    a settled sentence; the traceback stays in the log."""
    import voice_library
    logger.warning("voice %s failed unexpectedly: %s", what, str(exc)[:300])
    return HTTPException(
        status_code=502,
        detail={"code": "voice_engine_failed",
                "message": voice_library.ENGINE_FAILED_MESSAGE})


def _declared_length_exceeds(content_length: str | None, limit: int) -> bool:
    """True only for a parseable Content-Length above the limit. Absent or
    malformed headers decide nothing here; the body read still applies
    the hard limit afterwards."""
    try:
        return int(str(content_length or "").strip()) > limit
    except ValueError:
        return False


@app.post("/voices/custom/{slot}/sample",
          dependencies=[Depends(require_api_key)])
async def voices_upload(slot: str, request: Request, authorized: bool = False,
                        consent_version: str = "", source: str = "file"):
    """Store a custom voice sample (raw audio body, WAV/MP3/FLAC, bounded
    size) in the speech service's own store. No paths, no shell, no second
    sample store. An obviously oversized request is refused from its
    Content-Length before a byte is buffered; the post-read limit in
    voice_library stays, because Content-Length may be absent or wrong.
    Like every enrollment it needs the owner's acknowledgement
    (`authorized=true` and the `consent_version` they were shown)."""
    import voice_library
    if _declared_length_exceeds(request.headers.get("content-length"),
                                voice_library.MAX_SAMPLE_BYTES):
        raise HTTPException(
            status_code=413,
            detail=f"sample larger than "
                   f"{voice_library.MAX_SAMPLE_BYTES // (1024 * 1024)} MB")
    data = await request.body()
    try:
        return {"ok": True, **voice_library.upload_sample(
            slot, data, authorized=authorized,
            consent_version=consent_version, source=source)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001 - speech-service boundary
        raise _voice_unexpected(exc, "upload")


def _sample_too_large():
    import voice_library
    return HTTPException(
        status_code=413,
        detail={"code": "sample_too_large",
                "message": f"That file is larger than "
                           f"{voice_library.MAX_SAMPLE_BYTES // (1024 * 1024)} MB."})


@app.post("/voices/custom/create", dependencies=[Depends(require_api_key)])
async def voices_create(request: Request, name: str = "", note: str = "",
                        authorized: bool = False, consent_version: str = "",
                        scope: str = "dj", source: str = "file",
                        announcement_default: bool = False):
    """Create a custom voice from a recording: the enrollment as ONE call.
    Name, note, the owner's acknowledgement (`authorized`, and the
    `consent_version` of the sentence they were shown), what the voice is
    for (`scope`), how the sample arrived (`source`) and whether it becomes
    the default announcement voice ride the query; the recording is the
    raw body. Nothing else is accepted -- no URL, no path, no slot.

    No slot is named by the caller: which of this release's three slots
    the voice occupies is storage, and the customer never learns it. When
    cloning is unavailable the answer is a settled 503, never a provider
    error."""
    import voice_library
    if _declared_length_exceeds(request.headers.get("content-length"),
                                voice_library.MAX_SAMPLE_BYTES):
        raise _sample_too_large()
    data = await request.body()
    try:
        created = voice_library.create_voice(
            name, data, note, authorized=authorized,
            consent_version=consent_version, scope=scope, source=source,
            announcement_default=announcement_default)
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001 - speech-service boundary
        raise _voice_unexpected(exc, "create")
    # voice_id travels because the CALLER needs a handle to preview and
    # select with; no surface shows it to a person.
    return {"ok": True, **created}


@app.post("/voices/custom/{voice_id}/replace",
          dependencies=[Depends(require_api_key)])
async def voices_replace(voice_id: str, request: Request,
                         authorized: bool = False, consent_version: str = "",
                         scope: str | None = None, source: str = "file"):
    """Put a new recording under an existing custom voice. Same body and
    acknowledgement as create; the voice keeps its name, gets a new
    identity (so an announcement committed against the old recording
    falls back instead of speaking in this one) and, when `scope` is
    given, what it is allowed for is chosen again."""
    import voice_library
    if _declared_length_exceeds(request.headers.get("content-length"),
                                voice_library.MAX_SAMPLE_BYTES):
        raise _sample_too_large()
    data = await request.body()
    try:
        replaced = voice_library.replace_voice(
            voice_id, data, authorized=authorized,
            consent_version=consent_version, scope=scope or None,
            source=source)
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001 - speech-service boundary
        raise _voice_unexpected(exc, "replace")
    return {"ok": True, **replaced}


@app.post("/voices/announcement-default",
          dependencies=[Depends(require_api_key)])
def voices_announcement_default(req: VoiceAnnouncementDefaultRequest):
    """Which authorized voice reads an announcement whose event names none
    ("" = the station's DJ voice). A preference, never a grant: 422
    `voice_not_authorized` for a voice not allowed for announcements."""
    import voice_library
    try:
        return {"ok": True,
                **voice_library.set_announcement_default(req.voice_id)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.delete("/voices/custom/{slot}", dependencies=[Depends(require_api_key)])
def voices_delete(slot: str):
    import voice_library
    try:
        return {"ok": True, **voice_library.delete_sample(slot)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001
        raise _voice_unexpected(exc, "delete")


# --- Voice Library: recordings kept apart from the Custom slots ------------
# (voice_collection). Listing keeps the slots' current recordings first;
# nothing here reaches Music Assistant. Assigning is a slot replace, with
# the same acknowledgement and the same staged swap as the Booth's.

class VoiceCollectionRenameRequest(BaseModel):
    name: str


class VoiceCollectionArchiveRequest(BaseModel):
    archived: bool


class VoiceCollectionAssignRequest(BaseModel):
    slot: str
    authorized: bool = False
    consent_version: str = ""


class VoiceCollectionDeleteRequest(BaseModel):
    confirm: str = ""


def _collection_call(fn, what: str):
    import voice_library
    try:
        return {"ok": True, **fn()}
    except voice_library.VoiceError as exc:
        import voice_collection
        if isinstance(exc, voice_collection.VoiceInUse):
            raise HTTPException(status_code=409, detail={
                "code": exc.code, "message": str(exc)})
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001 - engine or disk boundary
        raise _voice_unexpected(exc, what)


@app.get("/voices/collection", dependencies=[Depends(require_api_key)])
def voices_collection():
    import voice_collection
    return _collection_call(voice_collection.listing, "library")


@app.get("/voices/collection/{entry_id}/audio",
         dependencies=[Depends(require_api_key)])
def voices_collection_audio(entry_id: str):
    import voice_collection
    import voice_library
    try:
        data, ctype = voice_collection.audio(entry_id)
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    return Response(content=data, media_type=ctype)


@app.post("/voices/collection/import", dependencies=[Depends(require_api_key)])
async def voices_collection_import(request: Request, name: str = "",
                                   authorized: bool = False,
                                   consent_version: str = "",
                                   source: str = "file"):
    """Add a recording (raw body) to the Voice Library. Same
    acknowledgement and audio checks as making a voice; no slot changes
    and nothing reaches the voice engine."""
    import voice_collection
    import voice_library
    if _declared_length_exceeds(request.headers.get("content-length"),
                                voice_library.MAX_SAMPLE_BYTES):
        raise _sample_too_large()
    data = await request.body()
    return _collection_call(lambda: voice_collection.import_recording(
        name, data, authorized=authorized, consent_version=consent_version,
        source=source), "library import")


@app.post("/voices/collection/{entry_id}/rename",
          dependencies=[Depends(require_api_key)])
def voices_collection_rename(entry_id: str, req: VoiceCollectionRenameRequest):
    import voice_collection
    return _collection_call(
        lambda: voice_collection.rename(entry_id, req.name), "library rename")


@app.post("/voices/collection/{entry_id}/archive",
          dependencies=[Depends(require_api_key)])
def voices_collection_archive(entry_id: str,
                              req: VoiceCollectionArchiveRequest):
    import voice_collection
    return _collection_call(
        lambda: voice_collection.set_archived(entry_id, req.archived),
        "library archive")


@app.post("/voices/collection/{entry_id}/assign",
          dependencies=[Depends(require_api_key)])
def voices_collection_assign(entry_id: str, req: VoiceCollectionAssignRequest):
    import voice_collection
    return _collection_call(lambda: voice_collection.assign(
        entry_id, req.slot, authorized=req.authorized,
        consent_version=req.consent_version), "library assign")


@app.post("/voices/collection/{entry_id}/delete",
          dependencies=[Depends(require_api_key)])
def voices_collection_delete(entry_id: str, req: VoiceCollectionDeleteRequest):
    import voice_collection
    return _collection_call(
        lambda: voice_collection.delete(entry_id, req.confirm),
        "library delete")


@app.post("/voices/{voice_id}/tuning", dependencies=[Depends(require_api_key)])
def voices_tuning(voice_id: str, req: VoiceTuningRequest):
    """Speed and sentence pause for one built-in voice. 422 for a cloned or
    unknown voice or a value outside the shelf's range."""
    import voice_library
    try:
        return {"ok": True, **voice_library.set_tuning(voice_id, req.speed,
                                                       req.pause)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.post("/voices/{voice_id}/scope", dependencies=[Depends(require_api_key)])
def voices_scope(voice_id: str, req: VoiceScopeRequest):
    """What a shelved voice is authorized for:
    "dj" (the default), "announcement" (a calendar event may name it for
    an announcement; never the DJ voice) or "both". This is the ONE way a
    voice becomes requestable from a calendar -- nothing an event says can
    grant it. 422 for an unknown voice, an unknown scope, or making the
    current DJ voice announcement-only."""
    import voice_library
    try:
        return {"ok": True, **voice_library.set_scope(voice_id, req.scope)}
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)


@app.post("/voices/preview", dependencies=[Depends(require_api_key)])
def voices_preview(req: VoicePreviewRequest):
    """One short line in the given voice -- the fixed phrase, or the
    owner's own `text` -- returned as audio for local UI playback. Never
    enqueued, never a break, never an announcement, never history. A
    built-in may be previewed at an unsaved speed/pause."""
    import voice_library
    try:
        audio = voice_library.preview(req.voice_id, speed=req.speed,
                                      pause=req.pause, text=req.text)
    except voice_library.VoiceError as exc:
        raise _voice_failure(exc)
    except Exception as exc:  # noqa: BLE001
        raise _voice_unexpected(exc, "preview")
    media = "audio/wav" if audio[:4] == b"RIFF" else "audio/mpeg"
    return Response(content=audio, media_type=media,
                    headers={"Cache-Control": "no-store"})


# --- Booth data surfaces (READ ONLY) -----------------------------------------

def _booth_queue_id() -> str:
    """The queue the Booth describes -- the SAME ownership order the
    observer and the DJ status headline follow (dj_engine._status_queue):
    the active listening session's queue first, then the selected output,
    then the active station. '' when nothing is playing anywhere."""
    import dj_engine
    import settings
    import station_store
    try:
        dj_cfg = settings.load()["dj"]
    except Exception:
        dj_cfg = {}
    queue_id = dj_engine._status_queue(dj_cfg)
    if queue_id:
        return queue_id
    station = station_store.get_active_station()
    return (station or {}).get("queue_id") or ""


@app.get("/now/playing", dependencies=[Depends(require_api_key)])
def now_playing_card():
    import now_playing
    queue_id = _booth_queue_id()
    if not queue_id:
        return {"ok": True, "playing": None}
    try:
        return {"ok": True, "playing": now_playing.snapshot(queue_id)}
    except Exception as exc:  # noqa: BLE001 - MA boundary
        return {"ok": False, "playing": None, "error": str(exc)[:200]}


@app.get("/now/lyrics", dependencies=[Depends(require_api_key)])
def now_lyrics():
    """On-demand lyrics for the current track (synced when MA has LRC,
    plain otherwise, honest none for programs/unknown). Deliberately NOT
    an entity attribute, NOT logged, NOT persisted anywhere."""
    import now_playing
    queue_id = _booth_queue_id()
    if not queue_id:
        return {"ok": True, "kind": "none"}
    try:
        return {"ok": True, **now_playing.lyrics(queue_id)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": "none", "error": str(exc)[:200]}


@app.get("/now/queue", dependencies=[Depends(require_api_key)])
def now_queue(recent: int = 5, upcoming: int = 10):
    """Recently played + up next around the live ordered position. READ
    ONLY by construction -- this route cannot write anything anywhere."""
    import now_playing
    queue_id = _booth_queue_id()
    if not queue_id:
        return {"ok": True, "recent": [], "upcoming": [], "current": None,
                "complete": False}
    try:
        return {"ok": True, **now_playing.queue_view(queue_id, recent,
                                                     upcoming)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "recent": [], "upcoming": [], "current": None,
                "complete": False, "error": str(exc)[:200]}


@app.get("/program/history", dependencies=[Depends(require_api_key)])
def program_history(limit: int = 10):
    """The last N AIRED ParlorWave program elements, newest first. Listener
    history: no tokens, no lifecycle states, no errors."""
    import now_playing
    try:
        return {"ok": True,
                "history": now_playing.program_history(limit)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "history": [], "error": str(exc)[:200]}


@app.get("/dj/breaks/{token}", dependencies=[Depends(require_api_key)])
def dj_break_state(token: str):
    """Diagnostics: one break's lifecycle row (no audio)."""
    import dj_engine
    brk = dj_engine.get_break(token)
    if brk is None:
        raise HTTPException(status_code=404)
    out = {k: v for k, v in brk.items() if k != "audio_path"}
    out["confirmation"] = dj_engine.airing_evidence(brk)
    return {"ok": True, "break": out}


@app.get("/dj/diagnostics/opportunities", dependencies=[Depends(require_api_key)])
def dj_diagnostics_opportunities(limit: int = Query(50, ge=1, le=200), before: str | None = None,
                                 outcome: str | None = None, reason: str | None = None,
                                 kind: str = "dj_break"):
    """Broadcast Diagnostics: compact DJ opportunity rows, newest first, read
    from the existing forensic stream (no drafts or scripts; see detail)."""
    import broadcast_diagnostics
    try:
        return broadcast_diagnostics.list_opportunities(limit, before, outcome, reason, kind)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)[:200])


@app.get("/dj/diagnostics/opportunities/{opportunity_id}", dependencies=[Depends(require_api_key)])
def dj_diagnostics_opportunity(opportunity_id: str):
    """Broadcast Diagnostics: one opportunity's recorded drafts, reasons,
    aired copy, lifecycle and obligations, by its public 16-hex id."""
    import broadcast_diagnostics
    out = broadcast_diagnostics.opportunity_detail(opportunity_id)
    if out is None:
        raise HTTPException(status_code=404)
    return out


class StationFromCurrentRequest(BaseModel):
    queue_id: str | None = None
    count: int = Field(40, ge=10, le=80)


@app.post("/station/from-current", dependencies=[Depends(require_api_key)])
def station_from_current(req: StationFromCurrentRequest):
    """📻 Start Radio From This Song. The current track keeps playing
    untouched; the FUTURE of the queue becomes: two more tracks by the
    current artist, then a broader grounded station seeded from the song +
    artist. Honest {ok: false, reason} on any condition where the queue
    cannot be changed safely -- playback is never interrupted as a recovery
    mechanism. The same operation will later back the voice command.

    An explicit queue_id is "Start Radio on output X": the id (any MA
    player id, verbatim) is validated against the live inventory, refused
    with active_station_elsewhere while another queue owns the active
    station, and on success recorded as the selected output with DJ speech
    permission for X. Omitting queue_id keeps today's behavior exactly."""
    import station_radio
    return station_radio.start_from_current(queue_id=req.queue_id,
                                            count=req.count)


class ManualBreakRequest(BaseModel):
    queue_id: str | None = None


def _manual_break_context(req_queue_id: str | None):
    """(queue_id, session, refusal|None) shared by Force and Cancel: the DJ
    must be enabled and an active session must exist on the target queue --
    manual break control against a silent station has no meaning."""
    import dj_engine
    try:
        dj = settings.load().get("dj") or {}
    except Exception:
        dj = {}
    if not dj.get("enabled"):
        return None, None, {"ok": False, "reason": "no_active_station"}
    # A manual break acts on the station that is RUNNING, so the target
    # comes from the shared control policy rather than settings order --
    # `outputs.enabled[0]` is an allow-list entry, not where the DJ is.
    import output_target
    picked = output_target.resolve(req_queue_id, require_playing=True)
    if not picked.get("ok"):
        reason = ("no_active_station"
                  if picked["reason"] == output_target.NOTHING_PLAYING
                  else picked["reason"])
        refusal = {"ok": False, "reason": reason}
        if picked.get("candidates"):
            refusal["candidates"] = picked["candidates"]
        return None, None, refusal
    queue_id = picked["queue_id"]
    c = dj_engine._connect()
    try:
        row = c.execute(
            "SELECT session_id, started_at FROM sessions WHERE "
            "status = 'active' AND queue_id = ? "
            "ORDER BY started_at DESC LIMIT 1", (queue_id,)).fetchone()
    finally:
        c.close()
    if not row:
        return None, None, {"ok": False, "reason": "no_active_station"}
    return queue_id, {"session_id": row[0], "started_at": row[1]}, None


def _manual_break_response(result: dict):
    """Success is 200; every truthful refusal is 409 with {ok, reason}."""
    if result.get("ok"):
        return result
    return JSONResponse(status_code=409, content=result)


class StationReplenishRequest(BaseModel):
    count: int = Field(default=12, ge=1, le=12)


@app.post("/station/replenish", dependencies=[Depends(require_api_key)])
def station_replenish(req: StationReplenishRequest):
    """Explicitly top up the CURRENT ACTIVE station's tail. Manual only.

    This is the same engine automatic replenishment will call -- there is no
    separate test-append shortcut -- but nothing invokes it on its own: no
    boundary hook, no worker, no timer, no setting. `option:"add"` reaches
    Music Assistant only when a person asks, which is what makes this safe to
    live-gate before any trigger exists.

    Deliberately NARROW. There is no queue_id, no station_id, no URI list and
    no weights: it operates on the one active station and nothing else, so it
    cannot become a general-purpose queue mutation API. `count` (1-12) exists
    only so a supervised gate can append two tracks without changing
    production tuning.

    Deliberately NOT gated on runway thresholds. Low-water and starvation
    belong to the future automatic eligibility engine; a person asking for
    more music is the eligibility here.

    At most ONE append per request, under every outcome -- including an
    ambiguous one, where the batch is left pending for reconciliation rather
    than retried."""
    import station_refill
    return station_refill.replenish(req.count)


@app.get("/station/replenish/eligibility",
         dependencies=[Depends(require_api_key)])
def station_replenish_eligibility():
    """Would an automatic refill fire right now, and why? READ ONLY.

    This runs the exact world-state gathering and policy evaluation a future
    automatic caller will use -- the same code, not a parallel description of
    it -- so the thresholds can be watched against live production before any
    trigger exists. It is the shadow gate for automatic replenishment.

    It cannot change anything: no claim, no selection, no URI resolution, no
    Music Assistant write. `eligible: true` here means only that policy would
    permit an attempt; nothing acts on it, and the explicit
    POST /station/replenish remains the sole route that can start an append."""
    import queue_ownership
    import station_eligibility
    import station_store
    decision = station_eligibility.evaluate()
    return {"ok": True, **decision.as_dict(),
            # Queue ownership, read from the local row only: this
            # endpoint stays incapable of writing to Music Assistant.
            "ownership": queue_ownership.status_for(
                station_store.get_active_station())}


@app.post("/dj/break/force", dependencies=[Depends(require_api_key)])
def dj_break_force(req: ManualBreakRequest):
    """DJ AFTER THIS SONG: a first-class manual intent pinned to the item
    playing right now, airing at its natural ending boundary through the
    unchanged render/gate/insert machinery. Never a transport action; every
    refusal is honest (boundary_unavailable, break_imminent,
    other_speech_in_flight, news blocks, queue_unreadable) and changes
    nothing."""
    import dj_engine
    import ma_queue
    queue_id, session, refusal = _manual_break_context(req.queue_id)
    if refusal:
        return _manual_break_response(refusal)
    try:
        state = ma_queue.queue_state(queue_id)
    except Exception:
        return _manual_break_response(
            {"ok": False, "reason": "queue_unreadable"})
    return _manual_break_response(
        dj_engine.force_break(queue_id, state, session))


@app.post("/dj/break/cancel", dependencies=[Depends(require_api_key)])
def dj_break_cancel(req: ManualBreakRequest):
    """Cancel Next Break: suppresses exactly one ordinary cadence
    opportunity (revoking an in-flight ordinary intent when one exists, in
    the same transaction as the suppression marker). queued/airing breaks
    answer too_late truthfully; news and event liners are untouchable;
    repeat calls are idempotent while a skip is pending."""
    import dj_engine
    queue_id, session, refusal = _manual_break_context(req.queue_id)
    if refusal:
        return _manual_break_response(refusal)
    return _manual_break_response(
        dj_engine.cancel_next_break(queue_id, session))


@app.get("/dj/status", dependencies=[Depends(require_api_key)])
def dj_status():
    """Read-only DJ state for the dashboard: user-facing status (off/idle/
    waiting/warming/preparing/ready/up_next/on_air), scheduler-truth
    next-break target and ETA estimate, current-track reaction state, and a
    one-item last-completed-break summary, plus compact durable station
    identity. Music Brain is authoritative; HA never computes scheduler state
    independently. No side effects."""
    import dj_engine
    import queue_ownership
    import station_store
    import station_lock
    station = station_store.status_snapshot()
    active = station.get("active")
    if active:
        # Who programs the active station's queue, and what was done to
        # Music Assistant's per-queue Autoplay flag. Local row only.
        active["ownership"] = queue_ownership.status_for(active)
    return {"ok": True, "dj": dj_engine.status_snapshot(),
            # Can music stories air right now? The listener's preference AND
            # whatever else this service requires (today the rollout gate),
            # decided here so no client reads the gate itself.
            "stories_active": settings.dj_stories_active(),
            "service_started": SERVICE_STARTED,
            # Durable station identity. Read-only and compact: no
            # frozen profile, no witness list, no intended URIs. `preparing`
            # is non-null only while a Start Radio replacement is in flight or
            # stranded awaiting recovery.
            "station": {**station,
                        "mutation_in_flight": station_lock.is_held()}}


@app.get("/dj/stories", dependencies=[Depends(require_api_key)])
def dj_stories():
    """The DJ memory, read-only: both story switches (the listener's
    preference and the rollout gate, reported apart), what is prepared,
    and the last confirmed airings with their sources. No side effects --
    nothing here reads Music Assistant or a model."""
    import story_job
    out = {"ok": True, "enabled": settings.dj_stories_enabled(),
           "rollout": settings.dj_stories_rollout(),
           "active": settings.dj_stories_active()}
    if not story_job.memory_path().exists():
        # never prepared: report empty rather than create the file
        return {**out, "counts": {}, "recent_airings": []}
    memory = story_job.open_memory()
    try:
        out["counts"] = memory.counts()
        out["recent_airings"] = [
            {"story_id": a["story_id"], "artist": a["artist"],
             "kind": a["kind"], "aired_at": a["aired_at"],
             "telling": a["telling"],
             "source": {k: (memory.source_of(a["story_id"]) or {}).get(k)
                        for k in ("provider", "field", "ref")}}
            for a in memory.recent_airings(10)]
    finally:
        memory.close()
    return out


class StoryFeedback(BaseModel):
    verdict: str = Field(pattern="^(veto|retire|like)$")
    note: str = ""


@app.post("/dj/stories/{story_id}/feedback",
          dependencies=[Depends(require_api_key)])
def dj_story_feedback(story_id: str, req: StoryFeedback):
    """The owner's word on one story: veto (never again), retire (done
    with it) or like (keep it in rotation)."""
    import story_job
    memory = story_job.open_memory()
    try:
        if not memory.feedback(story_id, req.verdict, req.note):
            raise HTTPException(status_code=404, detail="unknown story")
    finally:
        memory.close()
    return {"ok": True, "story_id": story_id, "verdict": req.verdict}


@app.post("/dj/context", dependencies=[Depends(require_api_key)])
def dj_context_publish(payload: dict):
    """Home Assistant publishes the latest sanitized enrichment snapshot
    (weather / upcoming calendar events / one headline), independently of any
    break. Storing a snapshot NEVER plans, renders, enqueues, or touches
    transport -- the render-time topic editor reads it later. Strictly
    validated; malformed input is rejected with nothing stored."""
    import dj_context
    try:
        clean = dj_context.validate_snapshot(payload)
    except dj_context.ContextError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        snap = dj_context.store_snapshot(clean)
    except dj_context.LabClockMismatch as exc:
        # only ever under a lab clock offset; production never gets here
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "captured_at": snap["captured_at"],
            "sections": sorted(k for k in clean)}


class LabClockAdvanceRequest(BaseModel):
    seconds: float | None = None
    to: str | None = None        # a timezone-aware ISO instant


def _lab_clock_view() -> dict:
    import dj_context
    import service_clock
    out = service_clock.lab_status()
    local = dj_context.station_now()
    zone = dj_context.station_zone()
    out["station_now"] = local.isoformat(timespec="seconds") if local else None
    out["station_zone"] = getattr(zone, "key", None)
    return out


@app.get("/lab/clock", dependencies=[Depends(require_api_key)])
def lab_clock_status():
    """LAB ONLY: the service clock's real and simulated instants, its
    offset and the station's civil time. Answers 404 -- indistinguishable
    from no route -- unless the process was started with
    PARLORWAVE_LAB_CLOCK=1, and only after authentication."""
    import service_clock
    if not service_clock.lab_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    return _lab_clock_view()


@app.post("/lab/clock/advance", dependencies=[Depends(require_api_key)])
def lab_clock_advance(req: LabClockAdvanceRequest):
    """LAB ONLY: move the simulated service instant FORWARD, by `seconds`
    or to an aware instant `to`. Forward only and persisted; a reset is a
    restart with the lab data restored, never a step back. Moves nothing
    real: media, timeouts, locks, inference and TTS keep real time. 404
    unless PARLORWAVE_LAB_CLOCK=1 (after authentication)."""
    from datetime import datetime
    import service_clock
    if not service_clock.lab_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    if (req.seconds is None) == (req.to is None):
        raise HTTPException(status_code=422, detail="give seconds or to")
    seconds = req.seconds
    if req.to is not None:
        try:
            target = datetime.fromisoformat(req.to)
        except ValueError:
            raise HTTPException(status_code=422, detail="to is not ISO 8601")
        if target.tzinfo is None:
            raise HTTPException(status_code=422,
                                detail="to must carry a UTC offset")
        seconds = (target - service_clock.now()).total_seconds()
    try:
        service_clock.lab_advance(seconds)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _lab_clock_view()


@app.post("/announcements/observed", dependencies=[Depends(require_api_key)])
def announcements_observed(payload: dict):
    """Home Assistant reports the ANNOUNCE-prefixed calendar events it can
    currently see (plus each source calendar's health). Reconciling an
    observation NEVER plans, renders, enqueues, or touches transport --
    the scheduler pass reads the durable store later. Strictly validated;
    malformed input is rejected with nothing stored."""
    import announcements
    try:
        return announcements.observe(payload)
    except announcements.StoreUnavailable as exc:
        # fail closed: the untrusted store is neither read as empty nor
        # overwritten; the integration reports the failure as a status
        raise HTTPException(status_code=503,
                            detail=f"announcements store unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/announcements/status", dependencies=[Depends(require_api_key)])
def announcements_status():
    """The Station tab's Announcements card: settings as the engine reads
    them, the observation sync's freshness, upcoming occurrences with
    their lifecycle, and what recently aired or expired. Read-only.

    Payloads are in the response because the person reading the Station
    tab owns the calendar. They are never logged."""
    import announcements
    return announcements.status_snapshot()


# --- Station Programmer: Dial Stations + Test Station -------------------------
#
# CONFIGURATION AND DRY RUN ONLY. Nothing below reads or writes a queue,
# selects a player or issues a transport command: the preview reaches Music
# Assistant only through ma_catalog's allowlisted library listings, and the
# preset routes write station.dial and nothing else.

def _dial_view(cfg: dict) -> dict:
    import station_intent
    import station_resolver
    presets = cfg.get("presets") or []
    # Whether each preset's last test approves what the builder programs
    # NOW. Computed on every read and never written back: a preset tested
    # by an older builder keeps everything it stored.
    states = {p.get("id"): station_resolver.test_state(
                  p.get("last_test"), p.get("query") or p.get("source"))
              for p in presets if p.get("id")}
    # The station each preset starts NOW (term + approved draw), in the form
    # /station/live names the station on air: a retuned preset is a new
    # station even though its id is not.
    identities = {p.get("id"): station_resolver.preset_identity(p)
                  for p in presets if p.get("id")}
    return {"ok": True, "presets": presets,
            "builder_revision": station_resolver.STATION_BUILDER_REVISION,
            "test_states": states,
            "identities": identities,
            "problem": cfg.get("problem"),
            "max_presets": station_intent.MAX_PRESETS,
            "kinds": [{"kind": k, "buildable": v["buildable"],
                       "sources": list(v.get("sources") or ())}
                      for k, v in station_intent.KINDS.items()]}


@app.get("/station/dial", dependencies=[Depends(require_api_key)])
def get_station_dial():
    """The saved Dial Stations. A damaged block reads as no presets plus a
    `problem`, never as a guess."""
    return _dial_view(settings.dial_config())


@app.put("/station/dial", dependencies=[Depends(require_api_key)])
def put_station_dial(body: dict):
    """Replace the preset list: {"presets": [...]}. A preset without an id
    is new and is given one; ids are never reused. Saving a preset starts
    nothing."""
    try:
        return _dial_view(settings.save_dial_presets(
            (body or {}).get("presets")))
    except settings.SettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/station/preview", dependencies=[Depends(require_api_key)])
def post_station_preview(body: dict):
    """TEST STATION: build the station an intent describes, from the Music
    Assistant library, exactly as a real start would -- and change nothing.
    {"intent": {...}, "count": 25, "variant": 0}. The answer carries a
    `last_test` record a preset may store."""
    from datetime import datetime
    import station_resolver
    body = body or {}
    result = station_resolver.preview_station(
        body.get("intent"), count=body.get("count", 25),
        variant=body.get("variant", 0))
    if result.get("ok"):
        result["last_test"] = station_resolver.last_test_record(
            result, datetime.now().isoformat(timespec="seconds"))
    logger.info("station_preview origin=%s kind=%s status=%s strategy=%s "
                "tracks=%s artists=%s",
                (result.get("intent") or {}).get("origin"),
                (result.get("intent") or {}).get("kind"),
                result.get("status"),
                (result.get("resolution") or {}).get("strategy"),
                (result.get("summary") or {}).get("total_tracks"),
                (result.get("summary") or {}).get("unique_artists"))
    return result


# --- the Station Programmer on air (Mode 2) -------------------------------------
# Four routes, one engine (station_live). Two only read local state and Music
# Assistant's player list; two change a queue, and each names its effect:
# start appends a station's opening runway to ONE explicitly named, proven-
# isolated speaker; stop ends programming and sends nothing to Music
# Assistant at all.

@app.get("/station/live", dependencies=[Depends(require_api_key)])
def get_station_live():
    """Is a station on air, which, where, its runway at the last check, when
    it last refilled and why it last declined. Durable state only -- this
    route sends nothing to Music Assistant."""
    import station_live
    return station_live.status()


@app.get("/station/live/targets", dependencies=[Depends(require_api_key)])
def get_station_live_targets():
    """Every Music Assistant speaker with its isolation verdict: which ones a
    station may run on, and why the others may not. READ ONLY."""
    import station_live
    return station_live.targets()


@app.post("/station/live/start", dependencies=[Depends(require_api_key)])
def post_station_live_start(body: dict):
    """START (or switch) a station on ONE named speaker:
    {"target": <speaker id>, "preset_id": "preset_3"} or
    {"target": ..., "intent": {...}}, optional "variant", and for a preset
    optional "origin" (station_tab | dial). The Station tab, the Request
    Line and the Dial all start a station through this one route. The
    target is never inferred. Refused, with nothing changed, unless the
    speaker is proven isolated, music is playing or paused on it, and the
    station passes the minimum-viable rule. The house speaker is the one
    exception to isolation, by HANDOVER: the first start answers
    handover_required with what would be taken over and a token, and only
    {"handover": {"token": ..., "upcoming": "keep" | "replace"}} starts it.
    The first artist station after a restart reads the library's relations
    first, so it takes longer."""
    import station_live
    return station_live.start_station(body or {})


@app.post("/station/live/autoplay", dependencies=[Depends(require_api_key)])
def post_station_live_autoplay(body: dict):
    """The person's answer to a PENDING Autoplay restore after a house
    handover's Stop could not switch Autoplay back on:
    {"decision": "restore" | "dismiss"}, for the most recent station (only it
    can hold the question). The only way a restore is written after that
    Stop; refused when nothing is pending."""
    import station_live
    return station_live.decide_restore(str((body or {}).get("decision") or ""))


@app.post("/station/live/stop", dependencies=[Depends(require_api_key)])
def post_station_live_stop():
    """STOP programming. No refill is added again; the speaker keeps playing
    what is queued and its queue is not cleared. Its Autoplay stays off --
    unless the station took over the house speaker by a handover that found
    Autoplay on: then that one queue's Autoplay is turned back on."""
    import station_live
    return station_live.stop_station()


@app.get("/dj/context/calendar", dependencies=[Depends(require_api_key)])
def dj_context_calendar_diagnosis():
    """What Parlorwave can actually see of the listener's calendar.

    The read half POST /dj/context never had. Without it a calendar that
    was never published and a calendar whose every event sat outside the
    announcement window produced exactly the same silence, so a calendar
    that never reached a break could not be diagnosed. Read-only:
    it selects nothing and commits no suppression marks.

    Event summaries are in the response because the person who pressed
    Test Calendar owns the calendar. They are never logged.
    """
    import dj_context
    return {"ok": True,
            "calendar": dj_context.calendar_diagnosis(
                settings.get_section("dj"))}


@app.get("/dj/context/weather", dependencies=[Depends(require_api_key)])
def dj_context_weather_diagnosis():
    """What Parlorwave holds of the weather, and what a break would say.

    The read half of the weather publish, and the same shape the calendar
    route above has: it contacts no forecast service, selects nothing and
    commits no marks. It reports the station's own snapshot -- condition,
    temperature, the resolved high/low and the sentence a break started
    now would carry -- so "the DJ never mentions the weather" can be told
    apart from "no entity is chosen", "the publisher stopped" and "the
    reading is too old to state".
    """
    import dj_context
    return {"ok": True,
            "weather": dj_context.weather_diagnosis(
                settings.get_section("dj"))}


@app.get("/settings/playback", dependencies=[Depends(require_api_key)])
def get_playback_settings():
    return _get_settings_section("playback")


@app.put("/settings/playback", dependencies=[Depends(require_api_key)])
def put_playback_settings(patch: dict):
    return _put_settings_section("playback", patch)


@app.get("/observer/status", dependencies=[Depends(require_api_key)])
def observer_status():
    """Shadow MA-observer diagnostics (READ ONLY). Bounded by construction:
    a capped recent-events ring plus counters -- the comparison surface the
    dark live gate reads beside HA's own event stream."""
    import ma_observer
    return ma_observer.status()


# --- queue hygiene -----------------------------------------------------------
#
# These two routes are the whole manual surface: the smallest one that lets
# the engine (queue_hygiene.py) ship, be dry-run from inside the real
# process, and be driven BY HAND for one bounded maintenance pass under
# the real, process-local station mutation lock. Neither route schedules
# anything, and neither speaks to Music Assistant itself: the diagnostic
# calls dry_run, the manual pass calls run_pass, and the engine keeps every
# one of its gates. The automatic path lives in queue_hygiene_triggers,
# reached only from the observer; it is not routed through here.


class QueueHygienePassRequest(BaseModel):
    queue_id: str = Field(..., min_length=1)
    #: Explicit, and bounded by the engine's reviewed per-pass ceiling. Strict:
    #: "5", 5.0 and True are not a count an operator typed.
    max_deletes: StrictInt = Field(..., ge=1, le=queue_hygiene.MAX_DELETES_PER_PASS)


def _queue_hygiene_control_enabled() -> bool:
    """MUSICBRAIN_QUEUE_HYGIENE_CONTROL=1: the maintenance
    switch that makes the manual pass route exist. It is NOT the permission
    to mutate -- that is playback.queue_hygiene, which the engine checks on
    every pass -- and it never touches settings. Read from the process
    configuration, the same one the rest of this module was built from."""
    return _config.current().queue_hygiene_control


@app.get("/queue/hygiene", dependencies=[Depends(require_api_key)])
def queue_hygiene_diagnostic(queue_id: str = Query(..., min_length=1)):
    """READ ONLY: the engine's dry run over one explicit queue, inside the
    running process. Classifies the live future and reports what a pass
    WOULD do -- no delete, no lock, no meta write, no last_pass_at stamp, no
    settings change, no effect on the latch. `status` is the engine's own
    compact state (permission, latch record, last mutating pass, limits);
    `result` is the dry-run result exactly as the engine returns it. A thin
    adapter: nothing here classifies anything."""
    queue_id = queue_id.strip()
    if not queue_id:
        raise HTTPException(status_code=422, detail="queue_id is required")
    return {"ok": True, "queue_id": queue_id,
            "manual_control": _queue_hygiene_control_enabled(),
            "status": queue_hygiene.status(),
            "result": queue_hygiene.dry_run(queue_id)}


@app.post("/queue/hygiene/pass", dependencies=[Depends(require_api_key)])
def queue_hygiene_manual_pass(req: QueueHygienePassRequest):
    """MANUAL and BOUNDED: one real hygiene pass over one explicit queue, by
    hand. Answers 404 -- indistinguishable from no route -- unless
    MUSICBRAIN_QUEUE_HYGIENE_CONTROL=1, and only after authentication, which
    the switch never weakens. The engine keeps every gate: the
    playback.queue_hygiene permission, the safety latch, authority, the
    station mutation lock, the refill / Start Radio exclusions, the minimum
    gap, the playing state, the buffer, ownership and staleness, the serving
    belt and the protected-sequence check. This route sets max_deletes and
    nothing else; every other limit is the engine's reviewed default. `ok` means
    the route ran the engine; `outcome` is the engine's own word for what
    happened, unchanged."""
    if not _queue_hygiene_control_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    queue_id = req.queue_id.strip()
    if not queue_id:
        raise HTTPException(status_code=422, detail="queue_id is required")
    limits = queue_hygiene.Limits(max_deletes=req.max_deletes)
    logger.info("queue hygiene MANUAL pass requested: queue=%s max_deletes=%d",
                queue_id, req.max_deletes)
    result = queue_hygiene.run_pass(queue_id, limits=limits)
    return {"ok": True, "queue_id": queue_id, "limits": asdict(limits),
            "outcome": result["outcome"], "result": result}


# Providers whose players make sense as radio outputs: speaker platforms.
# TVs/consoles behind universal_player are audio-capable but clutter a
# phone-sized selector, so the set is curated: the speaker providers Music
# Assistant ships (chromecast players/groups, AirPlay, Sonos, Squeezelite,
# Snapcast, DLNA, WiiM, Bluesound, HEOS, MusicCast, Bose, Samsung, AmpliPi,
# MPD, the host's local audio, sync groups), the HA-satellite players Music
# Brain already serves, and MA's own Sendspin web player so a browser (e.g.
# a travelling laptop) can be a station output.
OUTPUT_PROVIDERS = ("hass_players", "chromecast", "sendspin", "airplay", "sonos",
                    "sonos_s1", "squeezelite", "snapcast", "dlna", "wiim",
                    "bluesound", "heos", "musiccast", "bose_soundtouch",
                    "samsung_wam", "amplipi", "mpd", "local_audio", "sync_group")


@app.get("/outputs", dependencies=[Depends(require_api_key)])
def outputs():
    """Available ParlorWave DJ outputs + the currently selected one (READ ONLY).

    The inventory comes straight from Music Assistant; the selection is the
    existing dj.outputs.default_request_output setting (falling back to the
    first enabled DJ output). Selection is WRITTEN through the existing
    PUT /settings/dj partial-patch route -- this endpoint deliberately adds
    no second write path and performs no mutation of any kind: reading it,
    or changing the selection, never touches transport, queues or playback.

    The configured/selected ids are always included in the list even when MA
    reports them unavailable at this moment (a sleeping speaker is still a
    real output), so the UI can render the truth instead of dropping the
    active zone."""
    import ma_queue
    import settings as _settings
    try:
        players = ma_queue.list_players()
    except Exception as exc:  # noqa: BLE001 - MA down = honest empty answer
        return {"ok": False, "reason": "ma_unreachable",
                "error": str(exc)[:200]}
    outs = _settings.load()["dj"]["outputs"]
    selected = outs.get("default_request_output") or (
        (outs.get("enabled") or [None])[0])
    keep_ids = set(outs.get("enabled") or [])
    if selected:
        keep_ids.add(selected)
    usable = [p for p in players
              if p.get("provider") in OUTPUT_PROVIDERS
              or p.get("player_id") in keep_ids]
    return {"ok": True, "selected": selected, "outputs": usable}


@app.put("/settings/bulletins", dependencies=[Depends(require_api_key)])
def put_bulletin_settings(patch: dict):
    """Persist the bulletin settings the News console edits.

    Writing settings is ONLY writing settings: it validates, saves through
    the same section machinery as dj/playback, and returns the effective
    document. It never fetches a feed, downloads audio, contacts Music
    Assistant or runs a scheduler pass -- pressing Save must not become a
    network action. Enabling live participation here only means a FUTURE
    ordinary scheduler tick may observe it.

    The one check beyond section validation: a mode this build cannot
    actually produce is refused rather than persisted. settings accepts
    both origins because either is a structurally valid document, but
    storing 'generated_briefing' today would leave the console showing a
    selected mode that every downstream path refuses."""
    import news_bulletins
    mode = (patch or {}).get("mode")
    if mode is not None and mode not in news_bulletins.IMPLEMENTED_MODES:
        raise HTTPException(
            status_code=422,
            detail=("bulletins.mode '%s' is not implemented yet; supported: %s"
                    % (mode, ", ".join(news_bulletins.IMPLEMENTED_MODES))))
    return _put_settings_section("bulletins", patch)


# --- News bulletins: MANUAL developer controls -------------------------------
# Every one of these runs only when a person presses something. There is no
# timer, no startup hook and no scheduler path into them, and none of them
# touches Music Assistant. `bulletins.enabled` gates FUTURE automatic
# station participation, so it deliberately does NOT gate test/fetch/preview.


@app.post("/news/feed/test", dependencies=[Depends(require_api_key)])
def news_feed_test(payload: dict | None = None):
    """Reachability + parse check for the configured (or supplied) feed.
    Writes nothing: no row, no audio, no settings."""
    import news_bulletins
    bulletins = (payload or {}).get("bulletins")
    return news_bulletins.test_feed(bulletins)


@app.post("/news/fetch", dependencies=[Depends(require_api_key)])
def news_fetch():
    """Fetch Latest: the proven downloader, invoked once, by hand. A second
    concurrent press is refused with 409 rather than queued."""
    import news_bulletins
    out = news_bulletins.fetch_latest_bulletin()
    if out.get("reason") == "download_in_progress":
        raise HTTPException(status_code=409, detail="download_in_progress")
    if out.get("reason") == "bulletin_on_air":
        # The cached bulletin is queued or playing. Replacing it now would
        # delete the file Music Assistant is about to read.
        raise HTTPException(status_code=409, detail="bulletin_on_air")
    if out.get("ok"):
        current = news_bulletins.current_bulletin() or {}
        out["state"] = current.get("state")
        out["source"] = current.get("source_label")
        out["display_title"] = current.get("display_title")
        out["audio_url"] = current.get("audio_url")
    return out


@app.get("/news/status", dependencies=[Depends(require_api_key)])
def news_status():
    """Bulletin state, twice: raw for diagnostics, and translated for the
    Newsroom.

    `news` is the engine's own compact state, unchanged -- the developer
    console and every existing consumer still read exactly what they read
    before. `newsroom` is the same facts as sentences a person can act on,
    produced by the view model beside the engine so the panel never has to
    own a copy of the lifecycle vocabulary.

    Read-only, both of them: no fetch, no scheduler pass, no writes."""
    import news_bulletins
    import newsroom
    status = news_bulletins.status()
    return {"ok": True, "news": status, "newsroom": newsroom.view(status)}


@app.get("/news/history", dependencies=[Depends(require_api_key)])
def news_history_recent(limit: int = 20):
    """What the Newsroom actually did, newest first.

    The question the screen could not answer before: a bulletin that was
    fetched, armed and then lost its boundary looked exactly like one that
    aired. Each row also says whether its audio is still on disk, asked
    for rather than remembered -- a Listen Again offered from memory is a
    button that fails when pressed.

    `latest` is the short list the Newsroom shows without being asked: the
    newest outcome for EACH source. `history` is everything behind it, which
    the screen keeps folded away -- the retention rule means most of those
    rows describe a recording that has been reclaimed, and a long list of
    things that can no longer be played is not what "Recent bulletins"
    should be.

    Read-only: no fetch, no scheduler pass, no writes."""
    import news_history
    return {"ok": True,
            "latest": news_history.latest_per_source(),
            "history": news_history.recent(limit=min(max(int(limit or 20), 1), 60))}


@app.get("/news/ui", dependencies=[Depends(require_api_key)])
def news_ui():
    """Disposable developer console for the bulletin plumbing. Deliberately
    plain: one HTML string, vanilla JS, no build step and no Home Assistant
    dependency.

    NOT A PRODUCT SURFACE. The Newsroom is the product surface; this is the
    bench underneath it, behind the API key, and nothing a customer can
    reach links to it. It survives because it can drive the registry and
    the raw lifecycle that the Newsroom deliberately does not expose."""
    return Response(content=_NEWS_UI_HTML, media_type="text/html")


_NEWS_UI_HTML = """<!doctype html>
<meta charset="utf-8"><title>ParlorWave Radio News Bulletins</title>
<style>
 body{font:15px/1.45 system-ui,sans-serif;background:#14161a;color:#e8e8ea;
      margin:0;padding:24px;max-width:760px}
 h1{font-size:19px;letter-spacing:.14em;text-transform:uppercase;margin:0 0 18px}
 fieldset{border:1px solid #2c3038;border-radius:8px;margin:0 0 16px;padding:14px}
 legend{padding:0 6px;color:#9aa3b2;font-size:12px;letter-spacing:.1em;
        text-transform:uppercase}
 label{display:block;margin:6px 0}
 input[type=text]{width:100%;padding:7px;background:#0e1013;color:#e8e8ea;
                  border:1px solid #2c3038;border-radius:5px;font-size:14px}
 button{background:#222831;color:#e8e8ea;border:1px solid #3a4150;
        border-radius:6px;padding:9px 15px;font-size:14px;cursor:pointer;
        margin:4px 6px 4px 0}
 button:hover:enabled{background:#2c3542}
 button:disabled{opacity:.45;cursor:not-allowed}
 .card{display:flex;gap:14px;align-items:center;background:#0e1013;
       border:1px solid #2c3038;border-radius:8px;padding:12px}
 .art{width:72px;height:72px;border-radius:6px;background:#1b1f26;
      display:flex;align-items:center;justify-content:center;font-size:28px;
      flex:0 0 auto}
 .art img{width:100%;height:100%;object-fit:cover;border-radius:6px}
 .t{font-weight:600}.s{color:#9aa3b2;font-size:13px}
 .muted{color:#767e8c}.err{color:#ff8f8f;white-space:pre-wrap}
 .ok{color:#8fd694}
 audio{width:100%;margin-top:10px}
 .note{color:#767e8c;font-size:12px;margin-top:4px}
</style>
<h1>ParlorWave Radio &mdash; News Bulletins</h1>

<fieldset><legend>Station participation</legend>
  <label><input type="checkbox" id="enabled"> Live / automatic participation</label>
  <div class="note">Off today, and nothing automatic exists yet. Test Feed,
    Fetch Latest and Preview work regardless of this switch.</div>
</fieldset>

<fieldset><legend>Bulletin mode</legend>
  <label><input type="radio" name="mode" value="publisher_audio"> Publisher audio</label>
  <label class="muted"><input type="radio" name="mode" value="generated_briefing"
    id="genmode" disabled> Generated briefing &mdash; coming later</label>
</fieldset>

<fieldset><legend>Publisher</legend>
  <div id="sources"></div>
  <label><input type="radio" name="source" value="custom"> Custom feed</label>
  <label>Custom feed URL
    <input type="text" id="customurl" placeholder="https://example.com/feed.xml">
  </label>
  <button id="save">Save settings</button>
</fieldset>

<fieldset><legend>Actions</legend>
  <button id="test">Test Feed</button>
  <button id="fetch">Fetch Latest</button>
  <button id="refresh">Refresh Status</button>
  <div id="msg" class="s"></div>
</fieldset>

<fieldset><legend>Current bulletin</legend>
  <div id="current"></div>
</fieldset>

<fieldset><legend>Schedule (planning only)</legend>
  <div id="schedule"></div>
</fieldset>

<div class="note">Automatic :30 insertion: NOT IMPLEMENTED. The planner
below decides only whether an event <em>would</em> exist; nothing on this
page touches Music Assistant or station playback, and Preview plays in
this browser only.</div>

<script>
const $ = (id) => document.getElementById(id);
let busy = false, state = null;

function msg(text, cls) { $("msg").className = "s " + (cls || "");
                          $("msg").textContent = text; }

function esc(s) { return String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function secs(v) { if (!v && v !== 0) return "-";
  const m = Math.floor(v / 60), s = Math.round(v % 60);
  return m + ":" + String(s).padStart(2, "0"); }

function mb(v) { return (v || v === 0) ? (v / 1048576).toFixed(1) + " MB" : "-"; }

function render() {
  const n = state || {};
  $("enabled").checked = !!n.enabled;
  document.querySelectorAll("input[name=mode]").forEach(
    (r) => { r.checked = r.value === n.mode; });
  const pub = n.publisher || {};
  $("sources").innerHTML = (n.builtin_sources || []).map((s) =>
    '<label><input type="radio" name="source" value="' + esc(s.id) + '">'
    + esc(s.label) + "</label>").join("");
  document.querySelectorAll("input[name=source]").forEach(
    (r) => { r.checked = r.value === pub.source; });
  $("customurl").value = pub.custom_feed_url || "";
  $("fetch").disabled = busy || !!n.download_in_progress;
  $("test").disabled = busy;

  const c = n.current;
  if (!c) {
    $("current").innerHTML = '<div class="muted">No bulletin cached yet. '
      + "Press Fetch Latest.</div>";
  } else {
    const art = c.artwork
      ? '<img src="' + esc(c.artwork) + '" alt="">' : "\\u{1F4F0}";
    $("current").innerHTML =
      '<div class="card"><div class="art">' + art + "</div><div>"
      + '<div class="t">' + esc(c.display_title) + "</div>"
      + '<div class="s">' + esc(c.display_artist) + "</div>"
      + '<div class="s">' + esc(c.item_title || "") + "</div>"
      + '<div class="s">published ' + esc(c.published_at || "-")
      + " &middot; " + secs(c.duration_s) + " &middot; " + mb(c.size_bytes)
      + " &middot; " + esc(c.state) + "</div></div></div>"
      + '<audio controls preload="none" src="' + esc(c.audio_url) + '"></audio>';
  }
  const s = n.schedule || {};
  const row = (k, v) => v == null || v === ""
    ? "" : '<div class="s">' + k + ": " + esc(v) + "</div>";
  $("schedule").innerHTML =
    '<div class="t">' + esc(s.automation || "unavailable") + "</div>"
    + row("target", s.target_at)
    + row("window", (s.window_start || "?") + " \\u2192 " + (s.window_end || "?"))
    + row("boundary item", s.intended_boundary_item_id)
    + row("estimated air", s.estimated_air_at)
    + row("bulletin", s.bulletin_token)
    + row("last result", s.last_result)
    + row("last completed hour", s.last_completed_target_hour)
    + row("blocked", s.duplicate_block_reason);
  if (n.last_error) { msg("last error: " + n.last_error, "err"); }
}

async function call(path, opts) {
  const r = await fetch(path, opts);
  let body = {};
  try { body = await r.json(); } catch (e) { body = {}; }
  return { status: r.status, body: body };
}

async function refresh() {
  const r = await call("/news/status");
  if (r.body && r.body.news) { state = r.body.news; render(); }
  return r;
}

$("refresh").onclick = async () => { msg("refreshing\\u2026"); await refresh();
                                     msg("status updated", "ok"); };

$("save").onclick = async () => {
  const source = (document.querySelector("input[name=source]:checked") || {}).value;
  const mode = (document.querySelector("input[name=mode]:checked") || {}).value;
  const patch = { enabled: $("enabled").checked, mode: mode || "publisher_audio",
                  publisher: { source: source || "custom",
                               custom_feed_url: $("customurl").value.trim() } };
  const r = await call("/settings/bulletins",
    { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch) });
  if (r.status >= 400) { msg("settings rejected: "
      + esc((r.body && r.body.detail) || r.status), "err"); return; }
  msg("settings saved", "ok");
  await refresh();
};

$("test").onclick = async () => {
  busy = true; render(); msg("testing feed\\u2026");
  const r = await call("/news/feed/test", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: "{}" });
  busy = false;
  const b = r.body || {};
  if (b.ok) {
    msg("feed ok \\u2014 " + esc(b.label) + ": \\u201C" + esc(b.item_title)
        + "\\u201D, published " + esc(b.published_at) + ", "
        + esc(b.content_type || "?") + (b.duration_s ? ", " + secs(b.duration_s) : ""),
        "ok");
  } else {
    msg("feed test failed: " + esc(b.reason || r.status)
        + (b.error ? " \\u2014 " + esc(b.error) : ""), "err");
  }
  render();
};

$("fetch").onclick = async () => {
  busy = true; render(); msg("fetching bulletin\\u2026 (this can take a few seconds)");
  const r = await call("/news/fetch", { method: "POST" });
  busy = false;
  const b = r.body || {};
  if (r.status === 409) { msg("a download is already running", "err"); }
  else if (b.ok && b.result === "already_current") {
    msg("already current \\u2014 nothing to download", "ok");
  } else if (b.ok) {
    msg("downloaded: \\u201C" + esc(b.title) + "\\u201D", "ok");
  } else {
    msg("fetch failed: " + esc(b.reason || r.status)
        + (b.error ? " \\u2014 " + esc(b.error) : ""), "err");
  }
  await refresh();
};

refresh();   // status only -- never triggers a feed or download
</script>
"""
