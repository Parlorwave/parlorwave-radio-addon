#!/usr/bin/env python3
"""Queue-native recurring DJ engine (v1).

The ONLY accepted DJ transport: a finite ParlorWave MP3 inserted as the NEXT
Music Assistant queue item, played by MA as ordinary media between songs.
This module never pauses, resumes, stops, skips, ducks, or otherwise touches
transport -- Music Assistant owns playback, and every failure path here ends
in "no break; music continues untouched".

Scheduling semantics (airtime-aware, v2): dj.programming.cadence_min_minutes
means the DESIRED MINUTES BETWEEN AUDIBLE BREAK STARTS. The anchor is the
previous break's air start (telemetry air_start; session start for the first
break of a session), the target airtime is anchor + cadence, and the engine
targets the first safe natural boundary whose track END lands at/after
(target - BOUNDARY_SLACK_S). Rendering is LATE-BOUND: the spoken text is
generated ~RENDER_LEAD_S before the estimated boundary with FRESH musical
context, and the recorded adjacency (intended_pre_* in telemetry) is
re-validated immediately before enqueue -- stale copy is revoked and
re-planned, never aired. Music Brain owns all timing deterministically; the
LLM receives computed facts and never decides due-ness. Time-remaining
estimates choose WHEN TO RENDER only; the insertion authority remains the
structural Flow gate (index_in_buffer == current_index), never a countdown.

Break lifecycle (dj_breaks table, sessions.db v7; restart-safe):

    planned    intent exists with a target airtime (DJ allowed for the queue,
               nothing else in flight); spoken text NOT yet frozen
    rendering  LLM text + TTS audio being produced (atomic claim, stamped
               with the claiming process: a claim whose process is gone
               is revoked at the next boot -- recover_orphaned_renders)
    ready      finite MP3 rendered, probed, servable; awaiting a boundary
    queued     inserted as NEXT on the MA queue (atomic claim = the
               duplicate-insertion guard: only one caller ever wins the
               ready->queued transition, across duplicate planner runs AND
               container restarts, because the claim is a conditional UPDATE
               on the persistent row)
    airing     the break's URL became the current MA item (observed via the
               ordinary track_start ingestion path)
    completed  a real music track started after airing
    missed     never aired (stale ready/queued past TTL; boundary never
               reopened) -- terminal, audio becomes 410
    revoked    withdrawn deliberately -- terminal, audio becomes 410
    failed     generation/render/enqueue failed -- terminal, music untouched

Airtime is NEVER inferred from HTTP fetches of the audio: MA probes audio
long before it airs. Airing is observed only from the player's own
track_start carrying the break token (media_ref.break_token_of).

Grounding (v1, deliberately conservative): music-only banter from facts
Music Brain actually holds -- current/recent artist+title and honest session
observations. No album/year/trivia fabrication; no weather/news/calendar
(deferred so the known topic-planner starvation issue stays out of the DJ
critical path); no next-track promise on adopted queues.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets as _secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import media_ref
import dj_callback
import dj_persona
import paths
import providers
import tts_router
import break_packet
import config as _config_boundary
import deterministic_writer
import dj_writer
import smart_dj
from inference import llm_adapter
import speech
import service_clock

logger = logging.getLogger("musicbrain")

# --- states -------------------------------------------------------------------

# Why a break exists (telemetry 'purpose', not a column -- same home as
# target_air_at and the adjacency claims). 'scheduled' is the ordinary
# cadence break and stays implicit for legacy rows; 'station_request' is
# the Start Radio acknowledgement liner, which shares the entire transport
# but is NOT programming: it never becomes the cadence anchor.
PURPOSE_SCHEDULED = "scheduled"
PURPOSE_MANUAL = "manual"
PURPOSE_STATION_REQUEST = "station_request"
PURPOSE_NEWS_INTRO = "news_intro"
PURPOSE_ANNOUNCEMENT = "announcement"
# Purposes that are EVENT LINERS, not programming. They answer a specific
# event -- a listener pressing Start Radio, a news bulletin arriving, a
# calendar occurrence coming due -- so they never move ordinary cadence: a
# break that was due before one of these stays due after it. Membership
# here is the single definition; nothing rewrites air_start stamps to fake
# the same effect.
CADENCE_EXEMPT_PURPOSES = (PURPOSE_STATION_REQUEST, PURPOSE_NEWS_INTRO,
                           PURPOSE_ANNOUNCEMENT)
# Ordinary DJ speech: it participates in cadence when (and only when) it
# actually airs. 'manual' is deliberately NOT cadence-exempt -- a forced
# break that completes becomes the anchor exactly like a scheduled one,
# while one that never airs moves nothing (the anchor reads completed rows).
ORDINARY_PURPOSES = (PURPOSE_SCHEDULED, PURPOSE_MANUAL)
# A manual break follows the item that was current when the listener asked.
# If that item disappears before insertion (skip, near-end press, queue
# edit) it may re-pin ONCE to the new current music item; a second
# invalidation terminates it. Requests never chase a skip spree.
MANUAL_RETARGET_MAX = 1
ACK_MAX_WORDS = 30
ANCHOR_SCAN_ROWS = 20


def break_purpose(tel: dict[str, Any] | None) -> str:
    """The persisted purpose of a dj_breaks row. Absence means an ordinary
    scheduled break: plan_break has never written a purpose key, so every
    historical (and current) planner row is purpose-less by design."""
    return (tel or {}).get("purpose") or PURPOSE_SCHEDULED


def is_ordinary_break(tel: dict[str, Any] | None) -> bool:
    """True for cadence-real DJ speech (scheduled or manual); False for
    event liners (station_request, news_intro). The single definition --
    cancel targeting and the cadence anchor both use it."""
    return break_purpose(tel) in ORDINARY_PURPOSES

NON_TERMINAL_STATES = ("planned", "rendering", "ready", "queued", "airing")
TERMINAL_STATES = ("completed", "missed", "revoked", "failed")
VALID_STATES = NON_TERMINAL_STATES + TERMINAL_STATES
# States whose audio the endpoint may serve. 'ready' is included because MA
# probes the URL at enqueue time, before the queued transition lands.
SERVABLE_STATES = ("ready", "queued", "airing")

# WHO IS RENDERING. A 'rendering' row is work held in ONE process's memory
# -- the text and the TTS audio being produced -- and nothing else can
# finish it. Every claim into 'rendering' stamps this process's owner into
# telemetry in the same conditional UPDATE (_claim_render). A row owned by
# any other process, or owned by nobody and older than this process (a
# build from before owners existed), was abandoned when its process died:
# recover_orphaned_renders revokes it at boot instead of letting it hold
# every break until the stale-break TTL (otherwise a restart mid-render
# silences the DJ for that whole TTL). A 'rendering' row is never
# servable and nothing inserts from it, so none of these ever reached MA.
PROCESS_RENDER_OWNER = _secrets.token_hex(8)
PROCESS_STARTED_AT = service_clock.legacy_now()
ORPHANED_RENDER = "orphaned_render"

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")

# Rendered break audio lives under the DATA root, beside the other ephemeral
# stores; one small MP3 per break, deleted when the row is purged.
def audio_dir() -> Path:
    return paths.BASE_DIR / "broadcast_audio"


# A prepared break that never airs must not live forever: past this age a
# ready/queued break is terminal 'missed'; its audio answers the retired-
# silence tombstone from that moment (program_audio: known tokens are never
# transport-fatal).
READY_TTL_MINUTES = float(os.environ.get("DJ_BREAK_TTL_MIN", "30"))

# Terminal break history is bounded: completed rows keep about
# a day of detail, other terminal rows a few hours, and a hard row cap holds
# even if time-based retirement never fires. This bounds Music Brain storage
# and makes expired audio 404 for NEW fetches; it makes no claim about audio
# Music Assistant has already fetched or cached.
RETAIN_COMPLETED_H = float(os.environ.get("DJ_BREAK_RETAIN_H", "24"))
RETAIN_TERMINAL_H = float(os.environ.get("DJ_BREAK_RETAIN_FAILED_H", "6"))
BREAK_ROW_CAP = int(os.environ.get("DJ_BREAK_ROW_CAP", "200"))

# --- airtime-aware scheduling knobs (internal, deliberately NOT customer
# settings; env-overridable for operations only) --------------------------------
#
# RENDER_LEAD_S: how close to the estimated target boundary the spoken text
# is generated. A render takes about ten seconds, so 150s keeps a wide safety
# margin while the musical context stays fresh, and the structural gate is
# still reliably open that far from a track's end.
RENDER_LEAD_S = float(os.environ.get("DJ_RENDER_LEAD_S", "150"))
# MIN_RENDER_LEAD_S: the least REAL track time -- Music Assistant's musical
# time (_remaining_s), never the service clock -- a position tick must see
# before it may START a scheduled break's render. RENDER_LEAD_S says when
# rendering may begin; this says when it is already too late. A render begun
# with less than this left is finished after the boundary it was written
# for, fails pre-air as head_changed, and is replanned just past the next
# boundary -- where, on a short track, the next tick is too late again,
# because the observer restarts its 30 s position cadence at every boundary
# (on a run of short tracks, every attempt misses the same way).
#
# 20 s sits well above a typical break-ready time (the writer call plus
# synthesis; the render figure above). The margin is for what a typical
# render does not cover: the tail beyond it (a refused Smart
# draft followed by the deterministic floor costs a second TTS), the pass's
# own work around the render (context enrichment, the post-render queue
# re-read, pre-air revalidation, the insert round-trip), and MA's position
# evidence, which steps and arrives on a ~5 s poll. Holding is cheap -- the
# intent renders at the next boundary with the whole track ahead -- while
# starting too late costs a writer call, a revocation and that same next
# boundary anyway, so the bound errs early. It bounds the BOUNDARY, not MA's
# Flow read-ahead: insert telemetry shows the insertion gate can already be
# closed anywhere from tens of seconds to over two minutes before a track's
# end, while inserts succeed at 0 s on others, so no fixed lead could bound
# the gate; a break that misses it stays READY on the existing gate_closed
# path, exactly as before. It must stay well under
# RENDER_LEAD_S minus the 30 s position cadence, so the first tick inside a
# long track's render window always clears it: ordinary long-track
# scheduling never meets this gate.
MIN_RENDER_LEAD_S = float(os.environ.get("DJ_MIN_RENDER_LEAD_S", "20"))
# BOUNDARY_SLACK_S: a boundary landing up to this many seconds BEFORE the
# target airtime may be taken -- airing slightly early beats waiting a whole
# extra track. "First appropriate safe boundary near/after the target."
BOUNDARY_SLACK_S = float(os.environ.get("DJ_BOUNDARY_SLACK_S", "75"))
# Position ticks are the render-window heartbeat. If none arrived within
# this window (HA position forwarding down), boundary hooks render as a
# degraded fallback -- correctly anchored, just less fresh. It doubles as
# the staleness bound for airtime prediction: an MA musical-time checkpoint
# older than this is not evidence about where the track is now.
TICK_LIVE_S = 120.0
# The furthest ahead estimated_air_at will predict. Not a timing fudge: a
# sanity ceiling. A break airs at the end of the current track, so a
# plausible estimate is bounded by one track length; beyond this the inputs
# are not describing a real upcoming boundary and the caller gets None
# (no clock) instead of a confident wrong minute.
AIR_ESTIMATE_MAX_S = float(os.environ.get("DJ_AIR_ESTIMATE_MAX_S", "900"))
# After a terminal 'failed' render, wait this long before planning another
# intent for the queue: a hard-failing LLM/TTS must not become a per-tick
# retry storm. Deliberate stale-context revocations are exempt.
FAILED_RETRY_COOLDOWN_S = 90.0

# Smart DJ planner (smart_dj): how long one pass's claim on the planner
# decision for a planned intent stays exclusive. Observer ticks arrive every
# few seconds while a model call can take up to DJ_LLM_TIMEOUT_S; without a
# claim two concurrent passes would both ask the model at the same boundary.
# The claim is telemetry on the planned row (restart-safe, no new table) and
# expires so a crashed pass never wedges the intent.
PLANNER_CLAIM_TTL_S = float(os.environ.get("DJ_PLANNER_CLAIM_TTL_S", "90"))

# Module-scope tick liveness (heuristic only, never transport authority).
_last_tick_monotonic: float | None = None

# At most one automatic-news worker per process. Held for the whole life of
# a news pass (which may include an LLM-free TTS render), so repeated
# playback hooks are DROPPED rather than queued behind it.
_news_worker = threading.Lock()
_announcements_worker = threading.Lock()
_station_worker = threading.Lock()


def _note_tick() -> None:
    global _last_tick_monotonic
    _last_tick_monotonic = time.monotonic()


def _ticks_alive() -> bool:
    return (_last_tick_monotonic is not None
            and (time.monotonic() - _last_tick_monotonic) <= TICK_LIVE_S)

# Default break word budget (the 'normal' break_length). dj.programming.
# break_length IS consumed: it picks the word budget and how many optional
# contextual topics a break may carry. TTS speed
# is never changed; length is a prompt/structure budget.
MAX_WORDS = 60

# break_length -> deterministic composition policy. 'topics' lists which
# contextual topic kinds the deterministic editor may even consider: a SHORT
# break stays a concise music link, but urgent/relevant calendar information
# may still appear in it.
BREAK_LENGTH_POLICY: dict[str, dict[str, Any]] = {
    "short":  {"max_words": 35, "max_topics": 1, "topics": ("calendar",)},
    "normal": {"max_words": 60, "max_topics": 1,
               "topics": ("calendar", "news", "weather")},
    "long":   {"max_words": 90, "max_topics": 2,
               "topics": ("calendar", "news", "weather")},
}


def _length_policy() -> dict[str, Any]:
    """The composition policy for the CURRENT break_length setting, read at
    render time (settings failures fall back to 'normal'). The topic budget
    is additionally capped by dj.programming.topics_per_break."""
    try:
        import settings
        prog = settings.load()["dj"]["programming"]
        policy = dict(BREAK_LENGTH_POLICY.get(prog.get("break_length"),
                                              BREAK_LENGTH_POLICY["normal"]))
        policy["max_topics"] = min(int(policy["max_topics"]),
                                   int(prog.get("topics_per_break") or 1))
        return policy
    except Exception:
        return dict(BREAK_LENGTH_POLICY["normal"])


def persona_in_force(dj_settings: dict[str, Any] | None = None
                     ) -> tuple[dict[str, Any], str]:
    """(personality profile, chattiness level) for the break being rendered
    NOW. The PROFILE is read fresh from settings at render time so a Studio
    change reaches the next break without a restart. The LEVEL is not read
    from anything: since persona and airtime were decoupled it is always
    dj_persona.PRODUCTION_CHATTINESS, for every personality and for every
    legacy persisted value. A settings failure degrades to the house voice
    at the same level -- never to silence or to a different fact set; the
    facts never come from here."""
    try:
        if dj_settings is None:
            import settings
            dj_settings = settings.load().get("dj") or {}
        per = dj_settings.get("persona") or {}
        return (dj_persona.get_profile(per.get("profile")),
                dj_persona.effective_chattiness(dj_settings))
    except Exception:
        return dj_persona.get_profile(None), dj_persona.DEFAULT_CHATTINESS

# --- LLM / TTS default bindings (fully injectable; tests never touch these) ---
# HTTP mechanics and endpoint defaults live in providers.py; this engine only
# decides WHAT to ask (model, prompt, voice) and interprets the reply.

# The DJ writer's model and budget now live with the inference seam (the
# `dj_writer` role). These names stay as aliases: one source of truth, the
# long-standing env contract untouched, and every existing reference valid.
DJ_LLM_MODEL = llm_adapter.DJ_WRITER_MODEL
DJ_LLM_TIMEOUT_S = llm_adapter.DJ_WRITER_TIMEOUT_S
# DEVELOPMENT OVERRIDE ONLY. The product source of truth for the DJ voice is
# the settings document: dj.persona.voice.id is Music Brain's stable LOGICAL
# voice profile (e.g. custom_1), and dj.persona.voice.refs carries the
# source-specific backend resolution (refs.chatterbox_voice = the Chatterbox
# asset name). No installation's asset name -- including a developer's
# own -- is ever baked into the product. Empty everywhere = the TTS
# service's own default voice.
DJ_TTS_VOICE = os.environ.get("DJ_TTS_VOICE", "")


def dj_voice() -> dict[str, Any]:
    """The DJ voice in force for the next render
    (voice_library.dj_voice_in_force): the owner's selection, re-checked
    against the voice registry as it is NOW, or the service default when
    that selection is gone, holds another recording, or is no longer
    allowed for the DJ. Never raises: a registry that cannot be read
    degrades to the refs-free service default path it always did."""
    try:
        import voice_library
        return voice_library.dj_voice_in_force()
    except Exception:
        return {"voice_id": "", "selected_voice_id": "", "refs": {},
                "reason": None, "fallback_withheld": None}


def resolve_tts_voice() -> str:
    """Chatterbox asset name for the persona's LOGICAL voice profile.

    Resolution order: the in-force refs.chatterbox_voice (dj_voice: the
    settings selection, the one normal-production source of truth, minus
    a custom recording the DJ may no longer speak with) -> DJ_TTS_VOICE
    env (development override) -> '' (omit the voice; the service uses
    its default sample). Settings failures fall through rather than
    blocking a break."""
    ref = (dj_voice().get("refs") or {}).get("chatterbox_voice", "")
    if ref:
        return str(ref)
    return DJ_TTS_VOICE


def voice_identity() -> dict[str, str]:
    """Which voice this installation is about to speak with, as a stable
    LOGICAL fact: the persona's own voice id and the provider the router
    resolves it to.

    Break telemetry used to record only `resolve_tts_voice()`, which is a
    Chatterbox ASSET name. A Home Assistant TTS voice has no Chatterbox
    asset, so every break spoken in the base tier's only speech recorded an
    empty string, and `/dj/status -> last_break.voice` said nothing about a
    voice the system knew perfectly well. The logical id is what the voice
    library keys on and what a UI can turn into a name, so it is what gets
    written down. Never raises: telemetry must not be able to cost a break."""
    voice = dj_voice()
    voice_id = str(voice.get("voice_id") or "")
    if not voice_id:
        return {"id": "", "provider": ""}
    try:
        import tts_router
        provider = str(tts_router.decide(dict(voice.get("refs") or {}),
                                         resolve_tts_voice()).get("provider") or "")
    except Exception:
        provider = ""
    return {"id": voice_id, "provider": provider}


def voice_check_note() -> dict[str, Any] | None:
    """What a break row records when the DJ voice it speaks with is not
    simply the selection: the selection, and why it was set aside (or why
    its kept Chatterbox fallback was). None -- and nothing recorded --
    when the selection is in force whole. Bounded: two ids, two codes."""
    voice = dj_voice()
    if not (voice.get("reason") or voice.get("fallback_withheld")):
        return None
    return {"selected_voice_id": voice.get("selected_voice_id") or None,
            "reason": voice.get("reason"),
            "fallback_withheld": voice.get("fallback_withheld")}


def default_llm(prompt: str) -> str:
    """Grounded text for the dj_writer role, through the inference seam. The
    prompt's bytes, the model, the endpoint, the payload fields, the timeout
    and the failure posture are all exactly what they were before the adapter
    existed -- a missing 'response' degrades to '' (render_break turns empty
    text into a safe failed break) and every other failure propagates."""
    return llm_adapter.generate_text(llm_adapter.ROLE_DJ_WRITER, prompt)


def warm_llm() -> bool:
    """Best-effort warm-on-intent: load the model off the critical path.
    Never raises; a cold model just makes generation slower, not broken."""
    return llm_adapter.default_adapter().warm(llm_adapter.ROLE_DJ_WRITER)


def default_tts(text: str) -> bytes:
    """Speech for the persona's resolved logical voice, through the TTS
    router. Which voice speaks stays a settings/persona decision, not
    provider plumbing; the router only decides WHICH provider can say it.

    With no curated Kokoro voice selected, or the Kokoro kill switch off,
    this is byte-for-byte the call this function always made:
    providers.chatterbox_synthesize(text, resolve_tts_voice()).

    The refs routed are the IN-FORCE ones (dj_voice), decided at this
    call: a voice chosen after audio was handed to Music Assistant is
    heard from the next render on, and a selection the DJ may no longer
    speak with is never synthesized."""
    return tts_router.synthesize(text, refs=dict(dj_voice().get("refs") or {}),
                                 chatterbox_voice=resolve_tts_voice())


# The pinned break-audio output contract (the accepted queue-native prototype
# format): mono, 24 kHz, 64 kbps CBR MP3. CBR matters because ffmpeg writing
# to a non-seekable pipe cannot backfill a Xing/VBR duration header -- with
# constant bitrate, duration derives exactly from size/bitrate, so the
# downstream probe (the authoritative gate, audio_probe) stays reliable.
#: What a break looks like in the Music Assistant queue and in the MP3 tag:
#: the station, never a person's name.
BREAK_ITEM_TITLE = "DJ break"
BREAK_ITEM_ARTIST = "ParlorWave Radio"

MP3_ARGS = ["-ac", "1", "-ar", "24000", "-codec:a", "libmp3lame",
            "-b:a", "64k", "-f", "mp3"]


#: The shortest finite program item ParlorWave hands Music Assistant, in
#: seconds of audio (speech + any trailing pad). In non-Flow playback
#: MA hands the player the item AFTER
#: the current one only once it has seen the current one start -- a 1 s
#: preload poll, then call_later(1) (MA 2.10.4 player_queues/
#: stream_feeder.py) -- and nothing re-attempts a handover that misses. An
#: item that ends inside that window leaves the queue stuck on it: the
#: station falls silent, either at once or after the following song plays
#: untracked. This is Music Assistant's behaviour, with or without ParlorWave.
#: 0 disables the floor.
MIN_PROGRAM_ITEM_S = float(os.environ.get("DJ_MIN_PROGRAM_ITEM_S", "5.0"))
#: MP3 framing makes a padded file land a few ms either side of the target;
#: only a file short by more than this is padded again after the probe.
MIN_ITEM_TOLERANCE_S = 0.1


def min_item_pad(spoken_s: float | None, pad_s: float = 0.0) -> float:
    """The trailing pad that brings an item of `spoken_s` seconds (plus the
    pad already asked for) up to MIN_PROGRAM_ITEM_S. Unknown length -> the
    requested pad unchanged: the floor never guesses."""
    if spoken_s is None or MIN_PROGRAM_ITEM_S <= 0:
        return pad_s
    try:
        need = MIN_PROGRAM_ITEM_S - float(spoken_s)
    except (TypeError, ValueError):
        return pad_s
    return max(pad_s, round(need, 3)) if need > 0 else pad_s


def _append_silence(mp3: bytes, seconds: float) -> bytes:
    """The same finite MP3 with `seconds` of digital silence appended.

    Used for the news intro so the DJ's voice does not collide with the
    bulletin's first syllable -- a breath, not a gap. ffmpeg's apad does
    the padding in one pass through the SAME pinned MP3 output contract as
    every other rendered break, so the result is still an ordinary
    probeable finite MP3. argv form only, never shell=True."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-af", f"apad=pad_dur={seconds:g}", *MP3_ARGS, "pipe:1"],
        input=mp3, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg pad failed: {proc.stderr[:200]!r}")
    return proc.stdout


def _is_mp3(audio: bytes) -> bool:
    """An ID3 header or an MPEG frame sync: already the break's format."""
    return bool(audio) and (audio[:3] == b"ID3" or (
        len(audio) > 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0))


def _wav_to_mp3(wav: bytes) -> bytes:
    """Finite MP3 from any ffmpeg-readable audio (WAV from Chatterbox, WAV/
    OGG/FLAC/... from a Home Assistant provider) via ffmpeg (present in the
    production image),
    with the output contract pinned explicitly (MP3_ARGS) rather than left
    to ffmpeg defaults. argv form only -- never shell=True. A failed
    transcode raises, which render_break converts into a safe 'failed'
    break -- never a broken enqueue."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         *MP3_ARGS, "pipe:1"],
        input=wav, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg transcode failed: {proc.stderr[:200]!r}")
    return proc.stdout


# --- airing evidence ----------------------------------------------------------
#
# What a row can PROVE about its audio, strongest first. Each is recorded by
# the thing that observed it; nothing here is inferred from a schedule:
#   completed    seen current (air_start) and music started after it
#   started      Music Assistant reported it current and playing (air_start)
#   delivered    the whole file was fetched from us -- MA prefetches, so this
#                proves delivery, never air
#   queued       inserted into MA's queue (enqueue stamp), nothing since
#   unconfirmed  handed to MA, then gone without a start observation
#   skipped      ended before it was ever handed to MA
#   failed       the render or the insert failed
# Test-only "witnessed" (the audio witness) is not a runtime fact.

EVIDENCE_ORDER = ("completed", "started", "delivered", "queued",
                  "unconfirmed", "skipped", "failed")


def note_served(token: str, size: int, db_path=None) -> None:
    """Record that `token`'s whole file was sent (first/last time, count).
    Called after the response body went out; best-effort like _note."""
    try:
        brk = get_break(token, db_path)
        if brk is None:
            return
        served = dict((brk.get("telemetry") or {}).get("served") or {})
        now = _iso(service_clock.legacy_now())
        served.setdefault("first_full", now)
        served["last_full"] = now
        served["full_count"] = int(served.get("full_count") or 0) + 1
        served["bytes"] = int(size)
        _note(token, {"served": served}, db_path)
    except Exception:  # noqa: BLE001 - evidence never costs the response
        logger.debug("served note failed for %s", token, exc_info=True)


def airing_evidence(brk: dict[str, Any] | None) -> str | None:
    """The strongest airing evidence a row holds (EVIDENCE_ORDER)."""
    if not brk:
        return None
    tel = brk.get("telemetry") or {}
    state = brk.get("state")
    if tel.get("air_start"):
        return "completed" if state == "completed" else "started"
    if state == "failed" or tel.get("enqueue_failure"):
        return "failed"
    if tel.get("enqueue"):
        if state in ("missed", "revoked", "completed"):
            return "unconfirmed"
        return "delivered" if (tel.get("served") or {}).get("full_count") \
            else "queued"
    if state in TERMINAL_STATES:
        return "skipped"
    return None


# --- persistence helpers ------------------------------------------------------

def _connect(db_path=None):
    import playback
    playback.ensure_sessions_schema(db_path)
    return playback._connect(db_path)


def _iso(now: datetime | None = None) -> str:
    return (now or service_clock.legacy_now()).isoformat(timespec="seconds")


_COLS = ["token", "queue_id", "session_id", "state", "created_at",
         "updated_at", "spoken_text", "audio_path", "duration_s",
         "telemetry", "error"]


def active_breaks(queue_id: str, db_path=None) -> list[dict[str, Any]]:
    """Every NON-TERMINAL row on one queue, telemetry parsed, oldest
    first. A read for arbitration (news, announcements): who is holding
    or heading for the boundary right now."""
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT token, state, telemetry, created_at FROM dj_breaks "
            "WHERE queue_id = ? AND state IN ({}) ORDER BY created_at"
            .format(", ".join("?" * len(NON_TERMINAL_STATES))),
            (queue_id, *NON_TERMINAL_STATES)).fetchall()
    finally:
        c.close()
    out = []
    for token, state, tel_raw, created_at in rows:
        try:
            tel = json.loads(tel_raw or "{}") or {}
        except (TypeError, ValueError):
            tel = {}
        out.append({"token": token, "state": state, "telemetry": tel,
                    "created_at": created_at})
    return out


def get_break(token: str, db_path=None) -> dict[str, Any] | None:
    c = _connect(db_path)
    try:
        row = c.execute(f"SELECT {', '.join(_COLS)} FROM dj_breaks "
                        "WHERE token = ?", (token,)).fetchone()
    finally:
        c.close()
    if row is None:
        return None
    d = dict(zip(_COLS, row))
    try:
        d["telemetry"] = json.loads(d.get("telemetry") or "{}")
    except (TypeError, ValueError):
        d["telemetry"] = {}
    return d


#: The one terminal state that means A LISTENER HEARD THIS. Named here
#: because _transition is where the durable completion record is emitted.
STATE_COMPLETED = "completed"


def _transition(token: str, from_states: tuple[str, ...], to_state: str,
                now: datetime, db_path=None, **fields) -> bool:
    """Atomic state transition. Returns True only for the ONE caller whose
    conditional UPDATE matched -- the structural guard against duplicate
    planner runs and restart double-processing.

    THIS IS ALSO WHERE COMPLETION BECOMES DURABLE. Every authoritative
    move into `completed` goes through here, so exactly one forensic
    completion record is written per break, and the rowcount==1 guard
    above makes that true for free: a replayed or idempotent call matches
    no rows, returns False, and emits nothing.

    The gap this closes: the record used to be emitted from
    complete_airing alone, and news intros are completed by
    news_schedule through _transition instead, so the database could
    correctly say a break was heard while the durable stream had no record
    of it -- news intros silently missing from every question the stream
    could answer."""
    assert to_state in VALID_STATES
    sets = ["state = ?", "updated_at = ?"]
    args: list[Any] = [to_state, _iso(now)]
    for key, value in fields.items():
        sets.append(f"{key} = ?")
        args.append(value)
    marks = ",".join("?" for _ in from_states)
    args += [token, *from_states]
    c = _connect(db_path)
    try:
        cur = c.execute(
            f"UPDATE dj_breaks SET {', '.join(sets)} "
            f"WHERE token = ? AND state IN ({marks})", args)
        c.commit()
        won = cur.rowcount == 1
    finally:
        c.close()
    if won and to_state == STATE_COMPLETED:
        _note_coverage(token, now, db_path)
        _note_completed(token, now, db_path)
    if won and to_state in TERMINAL_STATES:
        _story_lifecycle(token, to_state, now, db_path)
    return won


# A break that completes this much SHORTER than its own audio was cut: its
# next item started before it could have finished (a skip, or Music
# Assistant resuming past it or before it after someone interrupted the
# queue). The observed span of an ordinary completed break (air_start ->
# completion, whole seconds, 1 Hz polls) can run a few seconds short of
# duration_s from polling slack alone; 6 s sits clear of that jitter.
AIRING_CUT_S = 6.0
COVERAGE_CUT = "cut"


def _note_coverage(token: str, now: datetime, db_path=None) -> None:
    """At completion: was the whole break on air? Recorded in telemetry
    `coverage` only when it provably was not (`cut`). The row still
    completes -- the audio gate needs a terminal state, and nothing is
    ever replayed -- but service_presence stops counting what it said as
    heard. Best-effort, like every telemetry write."""
    try:
        brk = get_break(token, db_path) or {}
        tel = brk.get("telemetry") or {}
        dur, started = brk.get("duration_s"), tel.get("air_start")
        if not dur or not started:
            return
        span = (now - datetime.fromisoformat(started)).total_seconds()
        if span < float(dur) - AIRING_CUT_S:
            _note(token, {"coverage": {"verdict": COVERAGE_CUT, "span_s": round(span, 1),
                                       "duration_s": round(float(dur), 1)}}, db_path)
            logger.info("dj break %s completed after %.1f s of %.1f s: cut",
                        token[:8], span, float(dur))
    except Exception:  # noqa: BLE001
        logger.debug("coverage check failed for %s", token, exc_info=True)


def _claim_render(token: str, now: datetime, db_path=None) -> bool:
    """planned -> rendering, with THIS process stamped as the render's
    owner in the same conditional UPDATE: a live render never exists
    without naming its owner. The atomic-claim contract is _transition's --
    True only for the one caller whose UPDATE matched."""
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE dj_breaks SET state = 'rendering', updated_at = ?, "
            "telemetry = json_set(COALESCE(telemetry, '{}'), "
            "'$.render_owner', ?) WHERE token = ? AND state = 'planned'",
            (_iso(now), PROCESS_RENDER_OWNER, token))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def _note_completed(token: str, now: datetime, db_path=None) -> None:
    """One durable completion record, after the commit and fenced:
    telemetry must never be able to undo or block the state transition it
    describes. The queue id is read back rather than passed, so every
    caller of _transition gets a complete record without having to know
    it is writing one."""
    queue_id = None
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT queue_id FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
        finally:
            c.close()
        queue_id = row[0] if row else None
    except Exception:  # noqa: BLE001 - a missing queue id is not worth a raise
        queue_id = None
    try:
        import dj_forensics as _fx
        _fx.completed(token, now, queue_id=queue_id)
    except Exception:  # noqa: BLE001
        logger.debug("forensic completed capture failed", exc_info=True)


def _stamp(token: str, key: str, now: datetime, db_path=None) -> None:
    """Record one telemetry timestamp. Best-effort by contract: telemetry
    must never block or fail the pipeline."""
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT telemetry FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
            if row is None:
                return
            try:
                tel = json.loads(row[0] or "{}")
            except (TypeError, ValueError):
                tel = {}
            tel[key] = _iso(now)
            c.execute("UPDATE dj_breaks SET telemetry = ? WHERE token = ?",
                      (json.dumps(tel), token))
            c.commit()
        finally:
            c.close()
    except Exception:
        logger.debug("telemetry stamp failed for %s/%s", token, key,
                     exc_info=True)


def _note(token: str, values: dict[str, Any], db_path=None) -> None:
    """Merge arbitrary keys into a break's telemetry JSON (target airtime,
    intended adjacency, ...). Best-effort like _stamp: telemetry must never
    block or fail the pipeline."""
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT telemetry FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
            if row is None:
                return
            try:
                tel = json.loads(row[0] or "{}")
            except (TypeError, ValueError):
                tel = {}
            tel.update(values)
            c.execute("UPDATE dj_breaks SET telemetry = ? WHERE token = ?",
                      (json.dumps(tel), token))
            c.commit()
        finally:
            c.close()
    except Exception:
        logger.debug("telemetry note failed for %s", token, exc_info=True)


# --- settings gates -----------------------------------------------------------

def dj_enabled_for_queue(queue_id: str) -> bool:
    """One definition: settings.dj_allowed_for_queue. Settings failures read
    as DISABLED -- the DJ must fail toward silence, never toward speech."""
    try:
        import settings
        return settings.dj_allowed_for_queue(queue_id)
    except Exception:
        return False


#: The one reason an ORDINARY DJ opportunity closes because DJ speech is
#: switched off (settings.dj.speech_enabled). It is a revocation reason,
#: never a failure: nothing was attempted, so it must never read as a
#: planner decline, a writer or TTS failure, or an insert failure, and it
#: never arms the failed-render retry cooldown.
SPEECH_DISABLED = "dj_speech_disabled"


def dj_speech_enabled() -> bool:
    """One definition: settings.dj_speech_enabled. Asked only about
    ORDINARY DJ hosting (is_ordinary_break) and the request liner -- never
    about news or announcements, which are station programming. Only an
    explicit false silences: anything unreadable resolves to the historical
    behaviour (speech), and dj_enabled_for_queue is what fails closed."""
    try:
        import settings
        return settings.dj_speech_enabled()
    except Exception:
        return True


def cadence_minutes() -> float:
    try:
        import settings
        return float(settings.load()["dj"]["programming"]["cadence_min_minutes"])
    except Exception:
        return 12.0


# --- planning -----------------------------------------------------------------

def last_air_anchor(queue_id: str, db_path=None) -> str | None:
    """ISO air-start of the newest completed SCHEDULED break for this queue:
    the telemetry air_start stamp (written when the break's URL became the
    current MA item), falling back to the completion timestamp for legacy
    rows that predate the stamp.

    Station-request acknowledgements are event liners, not programming:
    they are skipped here, so pressing Start Radio never resets ordinary
    cadence -- a break that was already due stays due afterwards. The scan
    is bounded rather than a single row, so a run of liners cannot hide the
    real anchor."""
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT telemetry, updated_at FROM dj_breaks WHERE queue_id = ? "
            "AND state = 'completed' ORDER BY updated_at DESC LIMIT ?",
            (queue_id, ANCHOR_SCAN_ROWS)).fetchall()
    finally:
        c.close()
    for tel_raw, updated_at in rows:
        try:
            tel = json.loads(tel_raw or "{}")
        except (TypeError, ValueError):
            tel = {}
        if not is_ordinary_break(tel):
            continue
        return tel.get("air_start") or updated_at
    return None


# --- manual skip marker (Cancel Next Break) -----------------------------------
# One queue-scoped row in the meta kv table: key dj_skip:{queue_id}, value
# JSON {suppressed_target, created_at, source, revoked_token}. It suppresses
# exactly ONE ordinary cadence opportunity and is deleted transactionally by
# the plan_break run that honors (or neutralizes) it. It never fakes an air
# anchor: the next real break simply targets one cadence later.

def _skip_key(queue_id: str) -> str:
    return f"dj_skip:{queue_id}"


def get_skip_marker(queue_id: str, db_path=None) -> dict[str, Any] | None:
    """Read-only view of a pending Cancel-Next-Break marker, or None."""
    c = _connect(db_path)
    try:
        row = c.execute("SELECT value FROM meta WHERE key = ?",
                        (_skip_key(queue_id),)).fetchone()
    finally:
        c.close()
    if not row:
        return None
    try:
        marker = json.loads(row[0])
        return marker if isinstance(marker, dict) else None
    except (TypeError, ValueError):
        return None


def render_pressure(queue_id: str, now: datetime | None = None,
                    horizon_s: float = 60.0, db_path=None) -> bool:
    """Is a spoken render in flight, or due within `horizon_s`, for this
    queue? The station programmer asks BEFORE borrowing the shared LLM:
    DJ/news deadlines outrank discretionary music ordering, so any pressure
    here means the advisor yields instantly (deterministic order ships)
    rather than queueing behind -- or in front of -- a deadline render.

    'rendering' is pressure by definition. A live intent ('planned' /
    'ready' / 'queued') is pressure when its render window (target minus
    RENDER_LEAD_S) opens inside the horizon -- 'ready'/'queued' rows keep
    counting as pressure because their AIRING is imminent and the next
    intent's render follows right behind. Rows without a target are
    boundary-opportunistic: treat them as pressure only in 'rendering'.
    Unreadable state is NOT pressure -- this guard protects speech, it
    never gets to silence music programming by breaking."""
    now_dt = now or service_clock.legacy_now()
    try:
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT state, json_extract(telemetry, '$.target_air_at'), "
                "json_extract(telemetry, "
                "'$.planner_declined.next_decision_not_before'), "
                "json_extract(telemetry, '$.purpose') "
                "FROM dj_breaks WHERE queue_id = ? AND state IN "
                f"({','.join('?' for _ in NON_TERMINAL_STATES)})",
                (queue_id, *NON_TERMINAL_STATES)).fetchall()
        finally:
            c.close()
        horizon_iso = _iso(service_clock.legacy_shift(now_dt, timedelta(
            seconds=horizon_s + RENDER_LEAD_S)))
        # DJ speech off: an ordinary intent that is planned or ready will be
        # CLOSED at its slot, never rendered or inserted, so it is no
        # deadline and must not keep the programmer's advisor idle. A render
        # actually in flight, and a break Music Assistant already holds,
        # still count.
        speech = dj_speech_enabled() if rows else True
        for state, target, not_before, purpose in rows:
            if state == "rendering":
                return True
            if (not speech and state in ("planned", "ready")
                    and (purpose or PURPOSE_SCHEDULED) in ORDINARY_PURPOSES):
                continue
            if target is not None and target <= horizon_iso:
                # Smart DJ: an intent the planner declined at the current
                # boundary cannot render again before the NEXT boundary
                # (the decline note records when that is, from the track
                # time the scheduler saw). Until that instant nears, the
                # due intent is not pressure -- silence must not keep the
                # programmer's advisor idle for the rest of the track.
                if (state == "planned" and not_before is not None
                        and not_before > horizon_iso):
                    continue
                return True
        return False
    except Exception:  # noqa: BLE001 - guard failure must not block music
        logger.warning("render_pressure unreadable; reporting none",
                       exc_info=True)
        return False



def plan_break(queue_id: str, session: dict[str, Any],
               now: datetime | None = None, db_path=None) -> str | None:
    """Create a break INTENT carrying its target airtime.

    Airtime-aware semantics: target_air_at = anchor + cadence, where the
    anchor is the later of session start and the previous break's AIR START
    (so cadence_min_minutes approximates minutes between audible break
    starts). The intent is created early -- it is cheap and freezes nothing;
    the render window, not intent creation, decides when spoken text is
    generated. Returns None (quietly) when: DJ is not enabled for the queue;
    another break for this queue is already in flight; or a render failed
    within FAILED_RETRY_COOLDOWN_S (storm guard -- deliberate stale-context
    revocations are exempt). Duplicate planner runs are structurally safe:
    the in-flight check plus the primary-key insert mean a second planner
    sees the first planner's row."""
    now_dt = now or service_clock.legacy_now()
    if not queue_id or not dj_enabled_for_queue(queue_id):
        return None
    c = _connect(db_path)
    try:
        inflight = c.execute(
            "SELECT COUNT(*) FROM dj_breaks WHERE queue_id = ? AND state IN "
            f"({','.join('?' for _ in NON_TERMINAL_STATES)})",
            (queue_id, *NON_TERMINAL_STATES)).fetchone()[0]
        if inflight:
            return None
        last_failed = c.execute(
            "SELECT MAX(updated_at) FROM dj_breaks WHERE queue_id = ? "
            "AND state = 'failed'", (queue_id,)).fetchone()[0]
    finally:
        c.close()
    if last_failed and last_failed > service_clock.cutoff_release(
            now_dt, timedelta(seconds=FAILED_RETRY_COOLDOWN_S)):
        return None  # a failing pipeline must not become a per-tick retry storm
    anchor = max(filter(None, [session.get("started_at"),
                               last_air_anchor(queue_id, db_path)]),
                 default=None)
    try:
        anchor_dt = datetime.fromisoformat(anchor) if anchor else now_dt
    except (TypeError, ValueError):
        anchor_dt = now_dt
    target_dt = service_clock.legacy_shift(
        anchor_dt, timedelta(minutes=cadence_minutes()))
    token = _secrets.token_hex(16)
    tel: dict[str, Any] = {"intent_created": _iso(now_dt)}
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        # Cancel-Next-Break marker: honored when this intent IS the
        # suppressed opportunity (skip one cadence slot), neutralized when
        # the opportunity it named never materialized (DJ was off through
        # it, session ended, cadence moved past it). Either way the marker
        # is consumed IN THIS TRANSACTION with the intent's creation, so a
        # crash can never leave a skip half-applied.
        row = c.execute("SELECT value FROM meta WHERE key = ?",
                        (_skip_key(queue_id),)).fetchone()
        if row:
            suppressed = None
            try:
                marker = json.loads(row[0])
                suppressed = datetime.fromisoformat(
                    marker["suppressed_target"])
            except (TypeError, ValueError, KeyError):
                pass  # unreadable marker: neutralize it below
            if suppressed is not None and target_dt <= suppressed + timedelta(
                    seconds=60):
                target_dt = suppressed + timedelta(
                    minutes=cadence_minutes())
                tel["skipped_opportunity"] = _iso(suppressed)
            c.execute("DELETE FROM meta WHERE key = ?",
                      (_skip_key(queue_id),))
        tel["target_air_at"] = _iso(target_dt)
        c.execute(
            "INSERT INTO dj_breaks (token, queue_id, session_id, state, "
            "created_at, updated_at, telemetry) VALUES (?, ?, ?, 'planned', ?, ?, ?)",
            (token, queue_id, session.get("session_id") or "", _iso(now_dt),
             _iso(now_dt), json.dumps(tel)))
        c.commit()
    finally:
        c.close()
    return token


def next_ordinary_opportunity(queue_id: str, session: dict[str, Any],
                              now: datetime | None = None,
                              db_path=None) -> datetime | None:
    """When the next ordinary DJ break that could still CARRY an
    announcement is due (server clock), or None when there is none to
    count on. Read-only; the same arithmetic plan_break uses (anchor +
    cadence, never earlier than now). A break already rendering or later
    is spoken for, and any other speech in flight, a pending Cancel Next
    Break or an unreadable store all mean "do not wait for the DJ"
    (an announcement caller then airs standalone)."""
    now_dt = now or service_clock.legacy_now()
    if get_skip_marker(queue_id, db_path) is not None:
        return None
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT state, telemetry FROM dj_breaks WHERE queue_id = ? AND "
            f"state IN ({','.join('?' for _ in NON_TERMINAL_STATES)})",
            (queue_id, *NON_TERMINAL_STATES)).fetchall()
    finally:
        c.close()
    for state, raw in rows:
        tel = json.loads(raw or "{}")
        if break_purpose(tel) != PURPOSE_SCHEDULED or state != "planned":
            return None
        target = datetime.fromisoformat(tel["target_air_at"])
        return max(target, now_dt)
    anchor = max(filter(None, [session.get("started_at"),
                               last_air_anchor(queue_id, db_path)]),
                 default=None)
    anchor_dt = datetime.fromisoformat(anchor) if anchor else now_dt
    target = service_clock.legacy_shift(
        anchor_dt, timedelta(minutes=cadence_minutes()))
    return max(target, now_dt)


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*(?:\s+|$)")


def weave_announcement(spoken: str, message: str,
                       context: dict[str, Any] | None) -> str:
    """The deterministic floor's words with the announcement read VERBATIM
    inside them: just before the sentence that introduces the next song
    (so the outro stays the last thing said), else at the end. The message
    is never trimmed, cleaned or guarded as DJ text."""
    msg = (message or "").strip()
    text = (spoken or "").strip()
    if not text:
        return msg
    if text[-1] not in ".!?":
        text += "."
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)
                 if m.group(0).strip()]
    music = _announcement_music(context)
    nxt = next((m for m in music if m.get("role") == "next"), None)
    names = [str(x).lower() for x in ((nxt or {}).get("artist"),
                                      (nxt or {}).get("title")) if x]
    for i in range(len(sentences) - 1, -1, -1):
        low = sentences[i].lower()
        if names and any(n in low for n in names):
            return " ".join(sentences[:i] + [msg] + sentences[i:])
    return f"{text} {msg}"


def _announcement_music(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The next song's names as the render context resolved them, in the
    packet shape smart_dj.announcement_failure reads."""
    post = str((context or {}).get("post_name") or "")
    if not post:
        return []
    import break_packet
    artist, _, title = post.partition(" - ")
    return [{"role": "next", "artist": artist.strip(),
             "title": break_packet.spoken_title(title)}]


def _announcement_room(context: dict[str, Any]) -> None:
    """A break carrying an announcement gets room for the message and
    sheds the optional enrichments that would stack on top of it
    (knowledge, album and character notes, semantic notes). Nothing that
    commits a mark is removed: required hosting -- clock, weather, the
    calendar, the songs -- stays exactly as owed."""
    import required_hosting
    msg = required_hosting.announcement_message(context)
    if not msg:
        return
    for key in ("knowledge_facts", "knowledge_reference", "album_run",
                "album_sequence", "ending_track_character",
                "next_track_character", "transition_character",
                "semantic_reason", "semantic_themes", "semantic_tone"):
        context.pop(key, None)
    if context.get("writer_style") == smart_dj.WRITER_P2:
        return      # P2's 112-word room already holds it; its prompt caps at 60
    extra = min(ANNOUNCEMENT_ROOM_MAX, len(msg.split()) + 6)
    context["max_words"] = int(context.get("max_words") or MAX_WORDS) + extra


# --- grounded content ---------------------------------------------------------

def _age_label(resolved_at: str | None, now: datetime) -> str | None:
    """Honest BUCKETED age of a finished track; None when the stamp is
    missing or unparseable, so the caller degrades to a bare name rather
    than inventing recency.

    A bounded set of human phrases the writer can repeat verbatim (an
    exact minute count invites paraphrase, and a paraphrase can drift to
    a different age; a phrase can be echoed as-is).
    The buckets are monotone and never overlap, so repeating the label
    exactly is always truthful."""
    if not resolved_at:
        return None
    # Spoken: only an EXACT age (a stamp from a repeated DST hour has two
    # and is left unsaid rather than guessed).
    seconds = service_clock.age_exact(resolved_at, now)
    if seconds is None:
        return None
    minutes = seconds / 60.0
    if minutes < 0:
        return None
    if minutes < 3:
        return "moments ago"
    if minutes < 10:
        return "a few minutes ago"
    if minutes < 20:
        return "about 15 minutes ago"
    if minutes < 40:
        return "about half an hour ago"
    if minutes < 75:
        return "about an hour ago"
    if minutes < 150:
        return "a couple of hours ago"
    if minutes < 300:
        return "a few hours ago"
    return "several hours ago"


def _request_ack_prompt(ctx: dict[str, Any]) -> str:
    """Kept for callers/tests: the ack prompt now lives in break_packet."""
    return break_packet.render_prompt(
        break_packet.build_packet(None, None, None,
                                  {**(ctx or {}),
                                   "purpose": PURPOSE_STATION_REQUEST}))


_character_line = break_packet.character_line


def build_break_prompt(session: dict[str, Any],
                       resolved: list[dict[str, Any]],
                       current: dict[str, Any] | None,
                       context: dict[str, Any] | None = None) -> str:
    """The Ollama writer's prompt: the grounded packet, rendered. The
    packet is the product's fact boundary (break_packet.build_packet); the
    rendering is the exact v2 prompt this function always produced
    (tests/fixtures/prompt_goldens pins it byte for byte)."""
    return break_packet.render_prompt(
        break_packet.build_packet(session, resolved, current, context))

# Clock closure (service wording polish). Only ever deletes: a bare
# part-of-day tag hung directly on a clock ("8:40 PM, evening.",
# "8:40 PM here, evening, and ...", "7:15 AM morning."), and a
# sentence-final "in the evening/morning/afternoon" after a clock that lost
# its meridiem, which becomes the supplied meridiem. Never touches "this
# evening", a sentence-initial "Overnight, ...", "overnight low", or any
# daypart not glued to a clock. Facts are untouched: the same clock, the
# same minute, one fewer dangling word.
_DAYPARTS_RE = r"(?:evening|overnight|afternoon|morning|night)"
_CLOCK_TOKEN_RE = r"\b\d{1,2}(?::\d{2})?\s*[AaPp]\.?[Mm]\.?"
_RE_BARE_TAG = re.compile(
    r"(" + _CLOCK_TOKEN_RE + r")\s*,?\s*(?:here,?\s*)?" + _DAYPARTS_RE
    + r"\s*(?=[.,!?;])")
_RE_IN_THE_TERMINAL = re.compile(
    r"\b(\d{1,2}:\d{2})\s+in the (evening|morning|afternoon)(?=[.!?])",
    re.IGNORECASE)


# A bare daylight recital -- the fact string spoken as a sentence of its
# own ("It is dark out.", "It's nighttime.") -- is deleted. Whole sentences
# only, so surrounding grammar is untouched; an integrated use ("It's dark
# out, so here is something bright.", "It's 7:15 AM here - still dark
# out.") is not a match and survives. Deletion only: the payoff, if any,
# belongs to the writer, never to this code.
_RE_BARE_DARK = re.compile(
    r"(?:(?<=^)|(?<=[.!?]\s))"
    r"It(?:['\u2019]s| is) (?:dark(?: out| outside| now| out there)?"
    r"|night ?time)[.!?]"
    # never orphan a follow-on that leans on it ("It's dark out. So lock
    # up."): the look-ahead spans the gap so it cannot be backtracked around
    r"(?!\s*(?:so|but|which|still|though|hence|that['\u2019]?s why|and)\b)"
    r"\s*",
    re.IGNORECASE)


def drop_bare_daylight_recital(text: str) -> str:
    out = _RE_BARE_DARK.sub("", text or "").strip()
    return out or (text or "")          # never empty a break into a failure


def close_clock_sentence(text: str) -> str:
    """Strong closure for a spoken clock: drop a bare daypart tag glued to
    it, and give a meridiem-less 'H:MM in the evening.' its meridiem."""
    out = _RE_BARE_TAG.sub(lambda m: m.group(1), text or "")
    out = _RE_IN_THE_TERMINAL.sub(
        lambda m: m.group(1) + (" AM" if m.group(2).lower() == "morning"
                                else " PM"), out)
    return re.sub(r"\s+([.,!?;])", r"\1", out)


def _mk_aired(spoken: str, context: dict[str, Any] | None) -> list[str]:
    """The knowledge fact_ids this copy realized -- the same test the
    novelty ledger uses, for the record only. Never raises."""
    try:
        ctx = context or {}
        if not ctx.get("knowledge_facts"):
            return []
        import music_knowledge as _mk
        return _mk.aired_fact_ids(
            ctx["knowledge_facts"],
            list(((ctx.get("smart_dj") or {}).get("fact_ids")) or []), spoken)
    except Exception:  # noqa: BLE001
        return []


def _mk_realized(spoken: str, context: dict[str, Any] | None) -> list[str]:
    """Knowledge this copy realized WITHOUT requiring a citation -- the
    deterministic floor's usage. For the record only; never spends
    novelty (music_knowledge.realized_fact_ids). Never raises."""
    try:
        ctx = context or {}
        if not ctx.get("knowledge_facts"):
            return []
        import music_knowledge as _mk
        return _mk.realized_fact_ids(ctx["knowledge_facts"], spoken)
    except Exception:  # noqa: BLE001
        return []


def _owed_obligations(context: dict[str, Any] | None,
                      budget: int) -> list[str]:
    """The required-hosting obligation set for this opportunity, whoever
    ends up satisfying it. Never raises: an unreadable obligation set
    reports nothing rather than costing the break."""
    try:
        import required_hosting
        return list(required_hosting.obligations(context, budget) or [])
    except Exception:  # noqa: BLE001
        return []


def _realized_topics(spoken: str, context: dict[str, Any] | None
                     ) -> list[str]:
    """Which contextual topics the FINAL COPY actually put on air.

    Air Check, the break list and the panel all read `context_topics`,
    and they are describing the break to a person who heard it -- so it
    has to mean 'this break said this', not 'this break was offered
    this'. A topic that only ever influenced the writer's thinking did
    not happen as far as the listener is concerned.

    Never raises: on any failure it claims nothing, which under-reports
    rather than inventing attribution."""
    out: list[str] = []
    try:
        ctx = context or {}
        text = spoken or ""
        if not text:
            return out
        import dj_context
        heard = dj_context.spoken_service_concepts(text, ctx)
        if "weather" in heard:
            out += [k for k in ("current_weather", "weather_context")
                    if ctx.get(k)]
        low = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        # The editor's fact strings quote the thing they are about
        # ("'Choir practice' starts at 8:40 PM (...)"), so the quoted phrase is
        # the anchor the copy had to realize.
        for key in ("upcoming_event", "news_headline"):
            fact = str(ctx.get(key) or "")
            m = re.search(r"['\"]([^'\"]{2,})['\"]", fact)
            anchor = m.group(1) if m else ""
            anchor = re.sub(r"[^a-z0-9]+", " ", anchor.lower()).strip()
            if anchor and anchor in low:
                out.append(key)
        # A knowledge fact counts as aired on exactly the terms the
        # novelty ledger uses: cited by the writer AND its subject
        # audible in the words that play (music_knowledge.aired_fact_ids).
        if ctx.get("knowledge_facts"):
            import music_knowledge as _mk
            cited = list(((ctx.get("smart_dj") or {}).get("fact_ids")) or [])
            if _mk.aired_fact_ids(ctx["knowledge_facts"], cited, text):
                out.append("knowledge_facts")
        # The planner's calendar is its own channel and is realized the
        # same way: by the event's name being audible.
        cal = ctx.get("calendar_event")
        if isinstance(cal, dict) and cal.get("title"):
            import smart_dj as _sdj
            if _sdj.calendar_spoken(text, {"calendar": cal}):
                out.append("calendar_event")
    except Exception:  # noqa: BLE001
        logger.warning("realized topic attribution failed (claiming none)",
                       exc_info=True)
        return []
    return out


def _clean_spoken(text: str, max_words: int = MAX_WORDS) -> str:
    text = re.sub(r"[\*_#`>\[\]]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip().strip('"\'')
    words = text.split()
    if len(words) > max_words * 2:
        text = " ".join(words[: max_words * 2])
    return drop_bare_daylight_recital(close_clock_sentence(text))


# --- rendering ----------------------------------------------------------------

def _resolve_callback(token: str, session: dict[str, Any],
                      resolved: list[dict[str, Any]] | None,
                      current: dict[str, Any] | None,
                      context: dict[str, Any] | None,
                      now_dt: datetime, db_path
                      ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(callback line or None, decision record).

    Ordinary music breaks only, cadence-gated, and gated on an
    EDITORIAL REASON: a documented relationship between the recalled
    artist and the music playing, or a shared title phrase. Recency is not
    a reason. The model plays no part: this runs before the prompt is even
    built. Fenced -- any failure means no callback, never a lost break.

    The decision record travels into telemetry and durable forensics even
    when nothing was recalled, because "why did the station not mention
    that?" is unanswerable from the aired copy alone."""
    decision: dict[str, Any] = {"considered": 0, "selected": False}
    try:
        ctx = context or {}
        brk = get_break(token, db_path)
        if not is_ordinary_break((brk or {}).get("telemetry")):
            # liners/news are not cadence and never carry a callback
            return None, {}
        chattiness = ctx.get("chattiness")
        if chattiness not in dj_persona.CHATTINESS_LEVELS:
            # Production level, not the profile's: callback depth is an
            # airtime decision and personality does not make it.
            chattiness = dj_persona.PRODUCTION_CHATTINESS
        pre = ctx.get("pre_name")
        if not pre and current and (current.get("title")
                                    or current.get("artist")):
            pre = (f"{current.get('artist') or 'Unknown'} - "
                   f"{current.get('title') or 'Unknown'}")
        state = callback_state(now_dt, (session or {}).get("started_at"),
                               db_path)
        if not dj_callback.eligible(chattiness, state):
            return None, {"considered": 0, "selected": False,
                          "suppression_reason": dj_callback.SUPPRESS_CADENCE}
        decision = dj_callback.select_candidate(
            resolved, lambda ts: _age_label(ts, now_dt),
            pre, ctx.get("post_name"), album_run=bool(ctx.get("album_run")),
            related_fn=_documented_relationship,
            told=_recent_callback_keys(now_dt))
        if not decision.get("selected"):
            return None, decision
        return dj_callback.sentence(decision), decision
    except Exception:  # noqa: BLE001
        logger.debug("earlier-session callback unavailable", exc_info=True)
        return None, decision


def _recent_callback_keys(now_dt: datetime) -> set[str]:
    """Callback stories told inside the cooldown. Fenced: unreadable state
    suppresses nothing, exactly like fact novelty."""
    try:
        import dj_context
        return dj_context.recent_callback_keys(now_dt)
    except Exception:  # noqa: BLE001
        return set()


def _documented_relationship(a: str, b: str) -> dict[str, Any] | None:
    """The AUTHORITATIVE link between these two artists, or None.

    The shipped Music Knowledge pack's own MusicBrainz relationship rows
    -- a direct relationship, or shared members -- as a record the
    callback can SAY (who played in what). Never a model, never a guess,
    and never the network. Fenced: no pack, no relationship, no
    callback."""
    try:
        import music_knowledge
        if not music_knowledge.available():
            return None
        return music_knowledge.artists_related(a, b)
    except Exception:  # noqa: BLE001
        return None


def _llm_copy(token: str, prompt: str, llm, budget: int,
              context: dict[str, Any] | None, wg_flags, db_path
              ) -> tuple[str | None, str | None]:
    """The LLM writer's attempt, exactly as it always ran: one draft, the
    authority guard and service-presence check, at most ONE repair
    rerender. Returns (spoken, None) on success or (None, reason) -- a
    provider error, empty output, or a final guard rejection -- for the
    caller to fall back on. Never raises for a writer failure."""
    import service_presence
    import writer_guard
    try:
        spoken = _clean_spoken(llm(prompt), max_words=budget)
    except Exception as exc:  # noqa: BLE001 - the provider boundary
        logger.warning("dj break %s: LLM writer failed (%s: %s)", token[:8],
                       type(exc).__name__, str(exc)[:160])
        return None, dj_writer.REASON_PROVIDER_ERROR
    if not spoken:
        logger.warning("dj break %s: LLM returned no usable text", token[:8])
        return None, dj_writer.REASON_EMPTY_OUTPUT
    # Post-writer authority guard. Generation succeeding does not make
    # the text safe to air (the audio sanity check's rule, applied to
    # words): the writer can invent musical relationships or station
    # habits from its pretrained knowledge of the NAMED songs even though
    # no gate put such an opportunity in the prompt. One repair rerender
    # with the same facts; a second violation hands the break to the
    # deterministic writer -- no break beats a bad break, and a grounded
    # handoff beats no break.
    try:
        _wg_cat = writer_guard.check(spoken, wg_flags) if wg_flags else None
    except Exception:  # noqa: BLE001
        logger.warning("writer authority guard failed open", exc_info=True)
        _wg_cat = None
    # Service presence: a due clock or weather line the draft left out
    # rides the SAME single repair rerender as an authority violation --
    # never a third model call -- and, unlike authority, never rejects the
    # copy: a coherent music handoff without the clock still airs, and
    # the ledger simply records that nothing aired.
    try:
        _missing = service_presence.omitted(spoken, context, budget)
    except Exception:  # noqa: BLE001
        logger.warning("service presence check failed open", exc_info=True)
        _missing = []
    if _wg_cat or _missing:
        instruction = ""
        if _wg_cat:
            instruction += writer_guard.repair_instruction(_wg_cat)
        if _missing:
            instruction += service_presence.repair_instruction(
                _missing, context)
        try:
            repaired = _clean_spoken(llm(prompt + instruction),
                                     max_words=budget)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dj break %s: LLM repair failed (%s: %s)",
                           token[:8], type(exc).__name__, str(exc)[:160])
            repaired = ""
        second = ((writer_guard.check(repaired, wg_flags)
                   if (repaired and wg_flags) else _wg_cat)
                  if _wg_cat else None)
        still = _missing
        if _missing:
            if repaired and not second:
                still = service_presence.omitted(repaired, context, budget)
            _note(token, {"service_presence": {
                "due": service_presence.due(context, budget),
                "omitted_initial": _missing, "repair_attempted": True,
                "omitted_final": still}}, db_path)
        if _wg_cat:
            guard_tel = {"violation_category": _wg_cat,
                         "initial_rejected": True,
                         "repair_attempted": True}
            if repaired and not second:
                spoken = repaired
                guard_tel["repair_passed"] = True
                _note(token, {"writer_authority_guard": guard_tel}, db_path)
            else:
                guard_tel.update(repair_passed=False, final_rejected=True,
                                 violation_category=second or _wg_cat)
                _note(token, {"writer_authority_guard": guard_tel}, db_path)
                logger.warning("dj break %s: writer authority violation "
                               "(%s) after one repair", token[:8],
                               second or _wg_cat)
                return None, dj_writer.REASON_GUARD_REJECTED
        elif repaired and not second and len(still) < len(_missing):
            spoken = repaired          # the repair supplied what was due
    return spoken, None


def _deterministic_copy(token: str, packet, budget: int,
                        context: dict[str, Any] | None, wg_flags,
                        reason: str | None, db_path
                        ) -> tuple[str, dict[str, Any]]:
    """Deterministic copy from the packet, checked ONCE by the same guard
    and presence ledger. Raises (-> the break fails with a precise reason)
    only when the packet has no track facts to write from.

    THE FLOOR IS THE FALLBACK AND MUST NOT BE LOST. A guard rejection here
    used to raise, and that can silence the same hosting slot twice: a
    song title that trips the move-noun lexicon through a masking gap gets
    the Smart draft AND its floor both refused, and nothing airs. Such a
    masking gap is fixed at source (writer_guard
    ._mask_forms), but the architecture must not depend on that fix being
    complete, because the two failure modes are not symmetric: a bad
    Smart draft has a floor beneath it, and a rejected floor has nothing.

    So a rejection of DETERMINISTIC copy is now treated as what this
    module already called it -- a defect in the guard, not unsafe copy --
    and the floor airs anyway, loudly recorded. That is sound because the
    floor's text is CLOSED: deterministic_writer fills fixed template
    shapes with listed packet facts, has no model, no persona voice, no
    trivia and, by its own contract, "no commentary on the move between
    songs" -- the very class this category names. It cannot state a sonic
    comparison, so a hit on it can only come from a supplied fact value.

    Nothing about the SMART path changes: a planner draft that trips the
    guard is still rejected and still falls here (_planner_copy and the
    free-text path above are untouched)."""
    import service_presence
    import writer_guard
    det = deterministic_writer.write(packet)
    if not det or not det.get("text"):
        raise RuntimeError(
            "deterministic writer: no track facts to write from"
            + (f" (after {reason})" if reason else ""))
    spoken = _clean_spoken(det["text"], max_words=budget)
    if not spoken:
        raise RuntimeError("deterministic writer produced no usable text")
    cat = None
    if wg_flags:
        try:
            cat = writer_guard.check(spoken, wg_flags)
        except Exception:  # noqa: BLE001
            logger.warning("writer authority guard failed open", exc_info=True)
    if cat:
        # Recorded as a DEFECT SIGNAL and overridden, never fatal: see the
        # docstring. The loud log line is the point -- a rejection here
        # means the guard matched something the closed templates cannot
        # have claimed, which is a masking or lexicon bug worth chasing.
        _note(token, {"deterministic_guard_rejected": cat,
                      "deterministic_guard_overridden": True}, db_path)
        logger.error(
            "dj break %s: authority guard flagged DETERMINISTIC copy (%s) -- "
            "airing it anyway and recording the flag. The floor's templates "
            "cannot make this class of claim, so this is a guard defect "
            "(masking or lexicon), not unsafe copy. Text: %r",
            token[:8], cat, spoken[:200])
    try:
        missing = service_presence.omitted(spoken, context, budget)
    except Exception:  # noqa: BLE001
        missing = []
    if missing:
        _note(token, {"service_presence": {
            "due": service_presence.due(context, budget),
            "omitted_initial": missing, "repair_attempted": False,
            "omitted_final": missing}}, db_path)
    return spoken, {"template_id": det.get("template_id"),
                    "required_facts_used": list(det.get("facts_used") or [])}


def _planner_copy(token: str, text: str, budget: int,
                  context: dict[str, Any] | None, wg_flags, db_path
                  ) -> tuple[str | None, str | None]:
    """The Smart DJ planner's validated words, given the same treatment a
    free-text draft gets AFTER its model call: the spoken-text cleaner, the
    authority guard once, and the service-presence ledger -- with no repair
    rerender, because the planner makes exactly one model call per
    boundary. (spoken, None) or (None, reason) like _llm_copy."""
    import service_presence
    import writer_guard
    spoken = _clean_spoken(text, max_words=budget)
    if not spoken:
        return None, dj_writer.REASON_EMPTY_OUTPUT
    cat = None
    if wg_flags:
        try:
            cat = writer_guard.check(spoken, wg_flags)
        except Exception:  # noqa: BLE001
            logger.warning("writer authority guard failed open", exc_info=True)
    if cat:
        _note(token, {"writer_authority_guard": {
            "violation_category": cat, "initial_rejected": True,
            "repair_attempted": False, "final_rejected": True}}, db_path)
        logger.warning("dj break %s: planner copy rejected by authority "
                       "guard (%s)", token[:8], cat)
        return None, dj_writer.REASON_GUARD_REJECTED
    try:
        missing = service_presence.omitted(spoken, context, budget)
    except Exception:  # noqa: BLE001
        missing = []
    if missing:
        _note(token, {"service_presence": {
            "due": service_presence.due(context, budget),
            "omitted_initial": missing, "repair_attempted": False,
            "omitted_final": missing}}, db_path)
    # DID THIS DRAFT ACTUALLY HOST THE SLOT? The copy is true and allowed
    # by here; the remaining question is whether it is radio. A draft can
    # pass every grounding guard and host nothing -- a bare weather line
    # with no music in it, or a short trivia fragment over a transition it
    # never mentions. Those are true, and they are not hosting.
    #
    # DELIBERATELY LOOSE. The model is not made to follow a script: the
    # OUTGOING track is not required and one music anchor is enough. (A
    # DUE clock is required: it is a station duty like weather, paid only
    # by the supplied time said exactly once.) A break that says "it's
    # 62 degrees, and here's Cold Orchard" is good radio
    # that never names what just ended, and replacing it with a template
    # would make the station worse.
    # `draft_shortfall` states exactly what is still refused.
    #
    # A shortfall is NOT a grounding failure and earns no repair call: it
    # only means this draft does not get to satisfy the slot, so the
    # deterministic floor hosts instead (render_break's `spoken is None`
    # branch, the same one a rejected draft has always taken).
    import required_hosting
    shortfall = required_hosting.draft_shortfall(spoken, context, budget)
    if shortfall and set(shortfall) <= required_hosting.COMPLETABLE:
        # COMPLETE, DON'T REPLACE. The draft is true, allowed
        # and hosted; it only skipped a station reading. The station adds
        # the reading in the floor's own words (required_hosting
        # .complete_service, which re-judges the result), the authority
        # guard sees the WHOLE text once more, and the host's words air.
        completed, how = required_hosting.complete_service(
            spoken, shortfall, context, budget)
        cat = None
        if completed and wg_flags:
            try:
                cat = writer_guard.check(completed, wg_flags)
            except Exception:  # noqa: BLE001
                logger.warning("writer authority guard failed open",
                               exc_info=True)
            if cat:
                how = {**how, "refused": f"guard:{cat}"}
                completed = None
        record = {"planner_reason": None, "draft_rejected": not completed,
                  "obligations": required_hosting.obligations(context, budget),
                  "unmet": [] if completed else shortfall,
                  "completion": {**how, "completed": bool(completed)}}
        _note(token, {"required_hosting": record}, db_path)
        if context is not None:
            context["required_hosting_record"] = record
        if completed:
            if missing:
                _note(token, {"service_presence": {
                    "due": service_presence.due(context, budget),
                    "omitted_initial": missing, "repair_attempted": False,
                    "completed": True, "omitted_final": []}}, db_path)
            logger.info("dj break %s: planner draft completed with the "
                        "station's own %s (%s)", token[:8],
                        ",".join(shortfall), how.get("position"))
            return completed, None
        logger.info("dj break %s: planner draft could not be completed "
                    "(%s); deterministic floor takes it", token[:8],
                    how.get("refused"))
        return None, dj_writer.REASON_HOSTING_INCOMPLETE
    if shortfall:
        record = {"planner_reason": None, "draft_rejected": True,
                  "obligations": required_hosting.obligations(context, budget),
                  "unmet": shortfall}
        _note(token, {"required_hosting": record}, db_path)
        # DURABLE, not only in break telemetry. A later review reads the
        # forensic stream; a shortfall that only ever appears on a
        # `dj_breaks` row is invisible to it, and the row is reaped
        # within hours (dj_forensics).
        if context is not None:
            context["required_hosting_record"] = record
        logger.info("dj break %s: planner draft did not host the slot "
                    "(%s); deterministic floor takes it",
                    token[:8], ",".join(shortfall))
        return None, dj_writer.REASON_HOSTING_INCOMPLETE
    return spoken, None


def _authority_flags(context: dict[str, Any] | None,
                     resolved: list[dict[str, Any]] | None):
    """The writer authority flags for this render, derived ONCE from
    the same keys the prompt builder used. Shared by the free-text path,
    the Smart DJ planner and the deterministic writer so every writer is
    held to one contract. None when the guard itself fails (fenced: a guard
    bug must never cost every break)."""
    try:
        import writer_guard
        import song_context as _sc
        return writer_guard.authority_flags(
            context, resolved, known_albums=_sc.known_album_titles(),
            known_artists=_sc.known_artist_names(),
            all_albums=_sc.known_album_titles_all())
    except Exception:  # noqa: BLE001
        logger.warning("writer authority guard failed open", exc_info=True)
        return None


def render_break(token: str, session: dict[str, Any],
                 resolved: list[dict[str, Any]],
                 current: dict[str, Any] | None,
                 llm_fn: Callable[[str], str] | None = None,
                 tts_fn: Callable[[str], bytes] | None = None,
                 transcode_fn: Callable[[bytes], bytes] | None = None,
                 context: dict[str, Any] | None = None,
                 now: datetime | None = None, db_path=None,
                 pad_fn: Callable[[bytes, float], bytes] | None = None,
                 render_deadline: float | None = None) -> bool:
    """planned -> rendering -> ready (or failed). Atomic claim first, so a
    duplicate invocation (second planner, restart) finds the state already
    taken and does nothing. Every failure lands in 'failed' with the cause
    recorded; nothing here can touch playback.

    render_deadline is the target boundary on time.monotonic(), when the
    caller measured one; it only decides whether a render the speech
    sanity guard rejected may be synthesized ONCE more (see
    _sanity_retry_refusal). None -- direct callers, liners, intros -- means
    no retry, exactly as before.

    context (late-bound musical adjacency, captured FRESH by the scheduler
    immediately before this call) is persisted into telemetry at claim time:
    intended_pre_* records which track this speech assumes it follows, and
    the scheduler re-validates that assumption against live MA state before
    any enqueue -- rendered copy whose adjacency no longer holds is revoked,
    never aired."""
    now_dt = now or service_clock.legacy_now()
    if not _claim_render(token, now_dt, db_path):
        return False
    context = dict(context or {})
    voice_evidence = None
    if "persona_profile" not in context or "chattiness" not in context:
        # Speech settings are read at render time (the scheduler normally
        # supplies them; liners and direct callers get the same resolution).
        profile, chattiness = persona_in_force()
        context.setdefault("persona_profile", profile["persona_id"])
        context.setdefault("chattiness", chattiness)
    if context:
        notes = {
            "intended_pre_name": context.get("pre_name"),
            "intended_pre_queue_item_id": context.get("pre_queue_item_id"),
            "intended_pre_index": context.get("pre_index"),
        }
        if context.get("post_name"):
            # The prompt is allowed to introduce next_track ONLY when it was
            # known at render time; record its queue identity so the future
            # claim is re-validated before enqueue exactly like the
            # predecessor claim.
            notes["intended_post_name"] = context.get("post_name")
            notes["intended_post_queue_item_id"] = context.get(
                "post_queue_item_id")
        offered_topics = [k for k in ("current_weather", "weather_context",
                                      "upcoming_event", "news_headline",
                                      "knowledge_facts")
                          if context.get(k)]
        if offered_topics:
            # What the WRITER WAS GIVEN. `context_topics` -- what the
            # listener actually heard -- is written after the copy
            # exists (_realized_topics); until then there is no honest
            # answer, and the old code's answer was this list, which is
            # why Air Check attributed `current_weather` to a break whose
            # text contained no weather at all.
            notes["context_offered"] = offered_topics
        if context.get("knowledge_diag") or context.get("knowledge_facts"):
            # WHAT THE PACK OFFERED, never what it holds: match status,
            # scopes, and the identity of each candidate fact. No source
            # records, no provenance payload, no object metadata.
            diag = dict(context.get("knowledge_diag") or {})
            diag["offered"] = [
                {"id": f.get("id"), "fact_id": f.get("fact_id"),
                 "subject_type": f.get("subject_type"),
                 "predicate": f.get("predicate"), "about": f.get("about")}
                for f in (context.get("knowledge_facts") or [])
            ][:smart_dj.MAX_KNOWLEDGE_FACTS]
            notes["knowledge"] = diag
        if context.get("estimated_air_at"):
            # The timing claim this break's clock fact was written against.
            # Comparing it with the observed air_start stamp is how clock
            # accuracy stays diagnosable after the fact.
            notes["estimated_air_at"] = context["estimated_air_at"]
        if context.get("smart_dj"):
            # The Smart DJ planner's decision this copy was written from:
            # kind, topic key, cited fact ids, packet size. Bounded, no
            # prompt, no model text, no reasoning (smart_dj).
            notes["planner"] = _planner_note(context["smart_dj"])
        # Which personality and talk budget this copy was written under:
        # the proof that a Studio change reached a specific break.
        notes["persona_profile"] = context.get("persona_profile")
        notes["chattiness"] = context.get("chattiness")
        # The voice this render resolves to; the same proof for a Studio
        # voice change. `tts_voice` stays the Chatterbox asset it always
        # was (existing rows carry it, '' = the service default); the
        # logical id and the resolved provider are recorded beside it,
        # because they are the only two that can describe every provider.
        notes["tts_voice"] = resolve_tts_voice()
        identity = voice_identity()
        notes["voice_id"] = identity["id"]
        notes["tts_provider"] = identity["provider"]
        check = voice_check_note()
        if check:
            notes["voice_check"] = check
        _note(token, notes, db_path)
        if context.get("purpose") != PURPOSE_ANNOUNCEMENT:
            # The same three facts, for the forensic record that outlives
            # this row. Not for an announcement: its voice is decided at
            # synthesis and has its own `announcement_voice` record.
            voice_evidence = {"voice_id": notes["voice_id"] or None,
                              "provider": notes["tts_provider"] or None}
            if check:
                voice_evidence["voice_check"] = check
    llm = llm_fn or default_llm
    use_llm = dj_writer.llm_in_use(llm_fn)
    tts = tts_fn or default_tts
    try:
        _stamp(token, "llm_generate", now_dt, db_path)
        budget = int((context or {}).get("max_words") or MAX_WORDS)
        # Earlier-session callback: resolved BEFORE and INDEPENDENTLY of
        # the writer, from verified session facts only. It is never shown to
        # the model in any form -- no identity, no sentence, no placeholder
        # -- so it cannot influence what the writer generates or repairs.
        callback, callback_decision = _resolve_callback(
            token, session, resolved, current, context, now_dt, db_path)
        if callback and (context or {}).get("announcement_required"):
            # an announcement break says the message and the hosting it
            # owes; an earlier-session story would only stack on top
            callback = None
            callback_decision = {**(callback_decision or {}),
                                 "selected": False,
                                 "suppression_reason": "announcement_break"}
        if callback:
            # the callback spends the break's words rather than extending it
            trimmed = dj_callback.writer_budget(budget, callback["text"])
            if trimmed is None:
                callback = None          # no room for a usable handoff
                callback_decision = {**callback_decision, "selected": False,
                                     "suppression_reason":
                                     dj_callback.SUPPRESS_DISPLACES}
            else:
                budget = trimmed
                if context and context.get("target_words"):
                    import dj_persona
                    policy = dj_persona.chattiness_policy(context.get("chattiness"))
                    lo_s, hi_s = policy.get("target_share", (0.40, 0.70))
                    context["target_words"] = (
                        max(1, int(round(budget * float(lo_s)))),
                        max(1, min(budget, int(round(budget * float(hi_s))))),
                    )
        if callback_decision:
            _note(token, {"callback_decision": callback_decision}, db_path)
            context["callback_decision"] = callback_decision
        # The grounded packet is the ONE input every writer gets; the
        # reduced budget must reach the WRITER, not just the trimmer.
        packet = break_packet.build_packet(
            session, resolved, current,
            {**(context or {}), "now": now_dt, "max_words": budget})
        prompt = break_packet.render_prompt(packet)
        # Post-writer authority guard flags, derived once from the
        # same keys the prompt builder used. Guard-internal errors are
        # fenced: a guard bug must never cost every break.
        # resolved supplies the earlier_this_session identities, so the
        # guard can scope contrast authority to the ADJACENT pair only
        # -- an earlier track's name in history never counts
        # as the ending/next track.
        _wg_flags = _authority_flags(context, resolved)
        import writer_guard
        spoken = None
        reason = None
        smart = context.get("smart_dj") or {}
        smart_draft = smart.get("draft")
        smart_accepted = None
        story_accepted = None
        if context.get("purpose") == PURPOSE_ANNOUNCEMENT:
            # OPERATOR PROGRAMMING. The copy arrives FINISHED (exact
            # operator wording, a validated Smart rewrite, or the
            # deterministic announcement renderer) and is carried
            # verbatim: no writer, no guard rewrite, no _clean_spoken
            # clock/daylight edits, no truncation. The only further
            # transformation is normalize_for_tts at synthesis, exactly
            # like every finite render. Empty copy fails the break --
            # an announcement must never air as silence or as generic
            # DJ copy.
            spoken = str(llm_fn("") if llm_fn else "").strip()
            if not spoken:
                raise RuntimeError("announcement copy is empty")
        elif context.get("required_hosting_only"):
            # The planner declined at enrichment level. The station still
            # owes this transition, so the deterministic floor below writes
            # it -- and NEITHER model path runs, because a decline must not
            # buy a second call at the free-text writer.
            reason = (context.get("planner_declined_reason")
                      or dj_writer.REASON_PLANNER_DECLINED)
        elif (context.get("story_break") or {}).get("text"):
            # Music-aware stories: words prepared and checked ahead of air
            # (dj_memory), assembled in code with the owed time/weather and
            # the handoff. They take the same cleaner, authority guard and
            # service-presence check as planner copy; a rejection hands the
            # break to the deterministic floor exactly as today.
            spoken, reason = _planner_copy(token, context["story_break"]["text"],
                                           max(budget, int(context["story_break"]
                                                           .get("words") or 0) + 5),
                                           context, _wg_flags, db_path)
            if spoken:
                story_accepted = spoken
            else:
                _story_rejected(token, reason, now, db_path)
        elif smart.get("text"):
            # Smart DJ: the planner already made its ONE model call and
            # validated the words against the packet and the authority
            # guard (smart_dj.validate). No second call, no repair: the
            # same guard runs once more against this render's flags, and a
            # rejection here hands the break to the deterministic writer
            # exactly as a rejected free-text draft always has.
            spoken, reason = _planner_copy(token, smart["text"], budget,
                                           context, _wg_flags, db_path)
            if spoken:
                smart_accepted = spoken
        elif use_llm:
            spoken, reason = _llm_copy(token, prompt, llm, budget, context,
                                       _wg_flags, db_path)
            if spoken:
                smart_accepted = spoken
        ann_msg = None
        if context.get("purpose") != PURPOSE_ANNOUNCEMENT:
            import required_hosting as _rh
            ann_msg = _rh.announcement_message(context)
        if ann_msg and spoken is not None:
            # Calendar announcements: whatever wrote these words, they
            # air as the announcement only if they carry all of it, and
            # not tacked on after the next song's introduction.
            failed = smart_dj.announcement_failure(
                spoken, {"announcement": {"message": ann_msg},
                         "music": _announcement_music(context)})
            if failed:
                _note(token, {"announcement_delivery": {
                    "mode": "refused", "reason": failed[0],
                    "detail": failed[1]}}, db_path)
                spoken, reason = None, failed[0]
                smart_accepted = None
        is_floor_aired = False
        shadow_text = None
        shadow_template_id = None
        shadow_facts_used = []
        if spoken is None:
            # Deterministic copy: primary with no LLM configured, the
            # fallback otherwise. ONE attempt, no repair, no recursion: its
            # sentences are the packet's own facts, so a guard rejection
            # here is a defect to fail loudly on, not to retry.
            if ann_msg:
                try:
                    spoken, det_tel = _deterministic_copy(
                        token, packet, budget, context, _wg_flags, reason,
                        db_path)
                except RuntimeError:
                    # no track facts: the message is the whole break
                    spoken, det_tel = "", {"template_id": None}
                spoken = weave_announcement(spoken, ann_msg, context)
                _note(token, {"announcement_delivery": {
                    "mode": "verbatim_floor", "reason": reason}}, db_path)
            else:
                spoken, det_tel = _deterministic_copy(token, packet, budget,
                                                      context, _wg_flags,
                                                      reason, db_path)
            is_floor_aired = True
            shadow_template_id = det_tel.get("template_id")
            shadow_facts_used = det_tel.get("required_facts_used") or []
            # PRIMARY means "this installation has no writer". A planner
            # that declined or whose draft was refused IS a writer that
            # failed, so the floor is a FALLBACK there even when no
            # free-text llm_fn is wired -- otherwise the telemetry would
            # say the station never had a writer at all.
            planner_failed = bool(smart.get("text")
                                  or context.get("required_hosting_only")
                                  or (context.get("story_break") or {}).get("text"))
            writer_tel = dj_writer.telemetry(
                dj_writer.PROVIDER_DETERMINISTIC,
                (dj_writer.MODE_DETERMINISTIC_FALLBACK
                 if (use_llm or planner_failed)
                 else dj_writer.MODE_DETERMINISTIC_PRIMARY),
                fallback_reason=reason, **det_tel)
        else:
            # Smart copy succeeded; evaluate non-airing shadow deterministic floor
            # for ordinary DJ hosting breaks (evaluation only, never mutates state)
            if (context or {}).get("purpose") in (None, "", "dj", "smart_dj"):
                try:
                    import deterministic_writer
                    det_shadow = deterministic_writer.write(packet)
                    if det_shadow and det_shadow.get("text"):
                        shadow_text = _clean_spoken(det_shadow["text"], max_words=budget)
                        shadow_template_id = det_shadow.get("template_id")
                        shadow_facts_used = list(det_shadow.get("facts_used") or [])
                except Exception:
                    logger.debug("shadow deterministic candidate failed", exc_info=True)
            if context.get("purpose") == PURPOSE_ANNOUNCEMENT:
                # Announcement provenance: EXACT copy is a template (fixed
                # words, no writer), a Smart rewrite came from the model, and
                # everything else is the deterministic announcement renderer.
                # Air Check answers "Exact or Smart or deterministic?" from
                # this note.
                _a_origin = str((context.get("announcement") or {})
                                .get("origin") or "exact")
                writer_tel = dj_writer.telemetry(
                    {"exact": dj_writer.PROVIDER_TEMPLATE,
                     "smart": dj_writer.PROVIDER_OLLAMA}.get(
                         _a_origin, dj_writer.PROVIDER_DETERMINISTIC),
                    f"announcement_{_a_origin}")
            elif context.get("purpose") == PURPOSE_NEWS_INTRO:
                # A TEMPLATE LINE, and the telemetry says so.
                # render_news_intro delivers its fixed text by feeding the
                # llm_fn seam a function that returns it -- no prompt is built
                # and no model is called, which is the whole point of a
                # template line. But `llm_in_use` treats any occupied seam as
                # an LLM, so this branch stamped `ollama / ollama_primary` on
                # an installation whose own health says no writer is
                # configured. The note exists to tell WHY copy came from
                # where; there it said the opposite of the truth, and anyone
                # reading it would go hunting for an external writer that does
                # not exist.
                writer_tel = dj_writer.telemetry(dj_writer.PROVIDER_TEMPLATE,
                                                 dj_writer.MODE_TEMPLATE_LINE)
            elif story_accepted:
                writer_tel = dj_writer.telemetry(
                    dj_writer.PROVIDER_TEMPLATE, dj_writer.MODE_STORY_PREPARED,
                    kind=(context.get("story_break") or {}).get("kind"),
                    required_facts_used=list(
                        (context.get("story_break") or {}).get(
                            "obligations_met") or []))
            elif smart.get("text"):
                writer_tel = dj_writer.telemetry(dj_writer.PROVIDER_OLLAMA,
                                                 dj_writer.MODE_SMART_PLANNER,
                                                 kind=smart.get("action"))
            else:
                writer_tel = dj_writer.telemetry(dj_writer.PROVIDER_OLLAMA,
                                                 dj_writer.MODE_OLLAMA_PRIMARY)
        _note(token, {"writer": writer_tel}, db_path)
        logger.info("dj break %s copy: %s%s", token[:8], writer_tel["mode"],
                    (" (" + str(writer_tel.get("fallback_reason")) + ")"
                     if writer_tel.get("fallback_reason") else ""))
        # The writer's own text has now passed the authority contract (one
        # repair at most). Join the deterministic callback and re-check the
        # WHOLE text once; a failure here fails the break closed -- no extra
        # model call, music untouched.
        if callback:
            # THE DESTINATION BELONGS TO ONE VOICE. The callback states an
            # OCCURRENCE and nothing else, and the artist coming up
            # is never a candidate, so there is no destination here to
            # collide with the writer's own handoff.
            spoken = dj_callback.compose(callback["text"], spoken)
            try:
                final_cat = writer_guard.check(
                    spoken, {**(_wg_flags or {}),
                             "authorized_spans": [callback["text"]]})
            except Exception:  # noqa: BLE001
                logger.warning("final composed check failed open",
                               exc_info=True)
                final_cat = None
            if final_cat:
                _note(token, {"earlier_callback_rejected": final_cat},
                      db_path)
                raise RuntimeError(
                    f"composed break authority violation ({final_cat})")
        if ann_msg and not is_floor_aired:
            _note(token, {"announcement_delivery": {
                "mode": "conversational"}}, db_path)
        # The break's topics as the LISTENER will experience them. This
        # is the attribution Air Check shows, so it is computed from the
        # accepted copy -- after the writer, the repair and the callback
        # composition -- and never from what the planner was handed.
        _note_atmosphere(token, context, spoken, story_accepted, now_dt,
                         db_path)
        if smart_accepted and (context or {}).get("atmosphere_colour"):
            _note_planner_colour(token, context, spoken, now_dt, db_path)
        realized = _realized_topics(spoken, context)
        _note(token, {"context_topics": realized}, db_path)
        try:
            import dj_forensics as _fx
            _fx.aired(token, spoken, context, writer_tel, realized,
                      [k for k in ("current_weather", "weather_context",
                                   "upcoming_event", "news_headline",
                                   "knowledge_facts") if context.get(k)],
                      _mk_aired(spoken, context), service_clock.legacy_now(),
                      hosting=context.get("required_hosting_record"),
                      callback=context.get("callback_decision"),
                      # Recorded for EVERY opportunity, including the ones
                      # the writer satisfied unaided -- those previously
                      # left no obligation record at all, which is exactly
                      # why the owed denominator could not be recovered.
                      owed=_owed_obligations(context, budget),
                      # What the copy realized without needing a citation:
                      # the deterministic floor never cites, so this is the
                      # only place its bridge usage shows up. Observability
                      # only; it does not spend novelty.
                      knowledge_realized=_mk_realized(spoken, context),
                      shadow_copy=shadow_text,
                      shadow_template_id=shadow_template_id,
                      shadow_facts_used=shadow_facts_used,
                      floor_aired=is_floor_aired,
                      callback_obj=callback,
                      smart_draft=smart_draft,
                      smart_accepted=smart_accepted,
                      voice=voice_evidence)
        except Exception:  # noqa: BLE001
            logger.debug("forensic aired capture failed", exc_info=True)
        _stamp(token, "tts_render", service_clock.legacy_now(), db_path)
        # The synthesizer gets a speech COPY (clock times spelled out so
        # they are read as times, not cardinals, and the station's own
        # frequency said the way radio says it); `spoken` -- what gets
        # stored, shown and quoted -- is untouched. Every finite render
        # (DJ breaks, news intros, announcements, request acks) passes here.
        try:
            import settings as _settings
            station_name = _settings.resolved_station_name()
        except Exception:  # noqa: BLE001 - speech polish is never load-bearing
            station_name = None
        # the station's own time, as the context supplied it: a bare clock
        # that is that time is said with its AM/PM, never a guessed one
        tts_text = speech.normalize_for_tts(
            spoken, station_name=station_name,
            station_clock=(context or {}).get("station_time"))
        synth_started = time.monotonic()
        audio = tts(tts_text)
        if not audio or len(audio) < 128:
            raise RuntimeError("TTS returned no usable audio")
        _stamp(token, "encode", service_clock.legacy_now(), db_path)
        if not _is_mp3(audio):
            # Chatterbox returns WAV, a Home Assistant provider whatever it
            # produces; the break contract is finite MP3 -- ours, normalized.
            audio = (transcode_fn or _wav_to_mp3)(audio)
        synth_s = time.monotonic() - synth_started
        # The synthesizer's success code is not proof the audio is
        # sane (dead air can sit inside a "successful" render).
        # Checked BEFORE the deliberate trailing pad and BEFORE any file
        # exists, so a rejected render leaves nothing servable and rides
        # the ordinary failure contract (state 'failed', retry cooldown,
        # music untouched).
        import audio_sanity
        sane, audio_stats = audio_sanity.check_speech(audio)
        if not sane:
            logger.warning(
                "break %s rejected: audio_sanity_failed reasons=%s "
                "duration_s=%s max_internal_silence_s=%s "
                "silence_fraction=%s audible_s=%s", token,
                ",".join(audio_stats.get("reasons") or []),
                audio_stats.get("duration_s"),
                audio_stats.get("max_internal_silence_s"),
                audio_stats.get("silence_fraction"),
                audio_stats.get("audible_s"))
            _note(token, {"audio_sanity": audio_stats}, db_path)
            first_reasons = ",".join(audio_stats.get("reasons") or ["unknown"])
            # The defect is the synthesizer's, not the copy's -- the
            # same text usually voices cleanly the next time. ONE more
            # synthesis of the same accepted text, only while it provably
            # fits before the boundary; never a new writer call, never a
            # second retry. Anything else rides the failure contract below.
            refusal = _sanity_retry_refusal(token, render_deadline, synth_s,
                                            db_path)
            if refusal:
                _note(token, {"audio_sanity_retry": {
                    "result": "skipped", "reason": refusal}}, db_path)
                raise RuntimeError("audio sanity failed: " + first_reasons)
            logger.info("break %s re-rendering once after audio_sanity "
                        "rejection (%.1f s to the boundary, synthesis took "
                        "%.1f s)", token, render_deadline - time.monotonic(),
                        synth_s)
            try:
                audio = tts(tts_text)
                if not audio or len(audio) < 128:
                    raise RuntimeError("TTS returned no usable audio")
                if not _is_mp3(audio):
                    audio = (transcode_fn or _wav_to_mp3)(audio)
            except Exception as exc:  # noqa: BLE001 - keep the first cause
                _note(token, {"audio_sanity_retry": {
                    "result": "error", "error": str(exc)[:150]}}, db_path)
                raise RuntimeError("audio sanity failed: " + first_reasons
                                   + " (re-render failed: "
                                   + str(exc)[:120] + ")") from exc
            sane, audio_stats = audio_sanity.check_speech(audio)
            _note(token, {"audio_sanity_retry": {
                "result": "passed" if sane else "failed",
                "stats": audio_stats}}, db_path)
            if not sane:
                logger.warning(
                    "break %s re-render rejected: audio_sanity_failed "
                    "reasons=%s duration_s=%s max_internal_silence_s=%s "
                    "silence_fraction=%s audible_s=%s", token,
                    ",".join(audio_stats.get("reasons") or []),
                    audio_stats.get("duration_s"),
                    audio_stats.get("max_internal_silence_s"),
                    audio_stats.get("silence_fraction"),
                    audio_stats.get("audible_s"))
                raise RuntimeError(
                    "audio sanity failed: " + first_reasons
                    + " (re-render: "
                    + ",".join(audio_stats.get("reasons") or ["unknown"])
                    + ")")
            logger.info("break %s re-render passed audio sanity", token)
        pad = float((context or {}).get("trailing_silence_s") or 0.0)
        spoken_s = (audio_stats or {}).get("duration_s")
        if spoken_s is None and MIN_PROGRAM_ITEM_S > 0:
            # the sanity pass failed open: measure the length once more
            # from the bytes, so the floor is still applied BEFORE the one
            # write (the post-probe pass below is the last resort)
            try:
                spoken_s = audio_sanity.analyze(audio).get("duration_s")
            except Exception:  # noqa: BLE001 - the probe below still guards
                spoken_s = None
        floor_pad = min_item_pad(spoken_s, pad)
        if floor_pad > pad:
            # THE PLAYER MUST NOT RUN DRY: too short an item ends before
            # Music Assistant hands the player the song after it, and MA
            # then never advances.
            _note(token, {"min_item_pad": {
                "spoken_s": spoken_s,
                "requested_pad_s": pad, "pad_s": round(floor_pad, 2),
                "min_item_s": MIN_PROGRAM_ITEM_S}}, db_path)
            pad = floor_pad
        if pad > 0:
            # Opt-in presentation shaping for OUR OWN artifact. Only the
            # news intro asks for it (so the DJ does not run straight into
            # the bulletin); ordinary breaks never set the key and are
            # byte-identical to before. A padding failure is not worth
            # losing the break over -- the unpadded audio still airs.
            try:
                audio = (pad_fn or _append_silence)(audio, pad)
            except Exception as exc:  # noqa: BLE001
                logger.warning("break %s silence pad failed (airing "
                               "unpadded): %s", token, exc)
        loudness_mark = None
        if _program_audio_as_sound_effect():
            # Music Assistant will not normalise a sound effect, so the
            # station levels the clip itself, here, before the token is
            # servable (program_loudness). Off: not a byte changes.
            audio, loudness_mark = _level_for_sound_effect(token, audio,
                                                           db_path)
        adir = audio_dir()
        adir.mkdir(parents=True, exist_ok=True)
        path = adir / f"{token}.mp3"
        # Written ONCE, already tagged. Writing the audio and then
        # rewriting the whole file to put a tag on it cost every break two
        # full writes of its audio for one file (a needless cost in storage
        # endurance); the tag is a byte string, so it goes in the same
        # write. Best effort like the tagger it replaces: if building the
        # tag fails the untagged audio still airs.
        path.write_bytes(_tagged(audio, loudness_mark))
        _stamp(token, "probe", service_clock.legacy_now(), db_path)
        duration = _probe_mp3(path)
        if not duration or duration <= 0:
            raise RuntimeError("rendered audio is not a probeable finite MP3")
        if (MIN_PROGRAM_ITEM_S > 0
                and duration < MIN_PROGRAM_ITEM_S - MIN_ITEM_TOLERANCE_S):
            # THE FLOOR, SECOND CHANCE: the speech analysis above can fail
            # open with no length (if an ffmpeg silence pass fails, a too-
            # short item would otherwise go to air unpadded). The written
            # file's own probe is authoritative; pad what it says is short.
            short_by = MIN_PROGRAM_ITEM_S - float(duration)
            audio = (pad_fn or _append_silence)(audio,
                                                round(short_by + 0.02, 3))
            path.write_bytes(_tagged(audio, loudness_mark))
            measured, duration = duration, _probe_mp3(path)
            if not duration or duration <= 0:
                raise RuntimeError("padded audio is not a probeable finite MP3")
            _note(token, {"min_item_pad": {
                "stage": "post_probe", "probed_s": measured,
                "pad_s": round(short_by + 0.02, 3), "padded_s": duration,
                "min_item_s": MIN_PROGRAM_ITEM_S}}, db_path)
        ok = _transition(token, ("rendering",), "ready", service_clock.legacy_now(),
                         db_path, spoken_text=spoken,
                         audio_path=str(path), duration_s=duration)
        if ok:
            _stamp(token, "ready_to_air", service_clock.legacy_now(), db_path)
            if callback:
                # recorded ONLY on a break that reached ready: a break that
                # failed earlier must never look like an aired callback
                _note(token, {"earlier_callback": {
                    "artist": callback["artist"],
                    "age_label": callback["age_label"],
                    "reason": callback.get("reason"),
                    "form_id": callback["form_id"],
                    "link_key": callback.get("link_key")}}, db_path)
                if callback.get("link_key"):
                    # the story is spent for CALLBACK_COOLDOWN_HOURS, and
                    # only now that a listener can actually hear it
                    try:
                        import dj_context
                        dj_context.commit_marks(
                            {"callbacks": [callback["link_key"]]},
                            service_clock.legacy_now())
                    except Exception:  # noqa: BLE001
                        logger.warning("callback cooldown commit failed",
                                       exc_info=True)
        return ok
    except Exception as exc:  # noqa: BLE001 - the failure contract boundary
        logger.warning("dj break %s render failed: %s", token, exc)
        _transition(token, ("rendering",), "failed", service_clock.legacy_now(), db_path,
                    error=str(exc)[:300])
        return False


def _tagged(audio: bytes, user_text: dict[str, str] | None = None) -> bytes:
    """The break's audio with title/artist so the queue UI identifies the
    item as the station, plus the loudness marker when it was levelled.
    Best effort: a missing tag never blocks a valid break (without the
    marker it is inserted as a track, which MA normalises)."""
    try:
        import audio_probe
        return audio_probe.tagged_bytes(audio, user_text,
                                        title=BREAK_ITEM_TITLE,
                                        artist=BREAK_ITEM_ARTIST)
    except Exception:                   # noqa: BLE001 - best effort by design
        logger.debug("ID3 tagging failed for a break", exc_info=True)
        return audio


def _program_audio_as_sound_effect() -> bool:
    try:
        import program_loudness
        return program_loudness.enabled()
    except Exception:  # noqa: BLE001 - the track path is the safe answer
        return False


def _level_for_sound_effect(token: str, audio: bytes,
                            db_path=None) -> tuple[bytes, dict[str, str] | None]:
    """(audio, marker) for a clip that will be inserted as a sound effect.
    Levelled: the raw levelled frames and the marker its tag must carry.
    Not levelled, for any reason: the clip exactly as it was and no marker,
    so it is inserted as a track and MA normalises it. The outcome goes in
    the break's telemetry under `loudness` either way."""
    import program_loudness as PL
    try:
        frames, report = PL.level(audio)
    except Exception as exc:  # noqa: BLE001 - unlevelled audio still airs
        _note(token, {"loudness": PL.log_outcome("break", token, None, exc)},
              db_path)
        return audio, None
    _note(token, {"loudness": PL.log_outcome("break", token, report)}, db_path)
    return frames, {PL.MARKER_DESC: PL.marker(report)}


def _probe_mp3(path: Path) -> float | None:
    """Finite-duration probe: the same validation MA's prefetch relies on."""
    import audio_probe
    return audio_probe.duration(path)


# --- insertion ----------------------------------------------------------------

# Where Music Assistant fetches a rendered break from: derived from the
# advertised service URL (config.py), never a built-in address.
AUDIO_URL_BASE = _config_boundary.current().audio_url_base


def audio_url(token: str) -> str:
    return f"{AUDIO_URL_BASE}/{token}.mp3"


# --- airtime-aware scheduling -------------------------------------------------
#
# Deterministic timing, owned entirely by Music Brain: wall-clock UTC-naive
# timestamps (consistent with every other store timestamp) for targets, MA's
# own elapsed_time for MUSICAL time, and monotonic time only for the local
# tick-liveness heuristic. The LLM never keeps or calculates time.

def _remaining_s(state: dict[str, Any],
                 wall_now: float | None = None) -> float | None:
    """Seconds until the current track's natural end, from MA's own
    elapsed_time (correct in Flow mode) plus wall drift since MA last
    updated it. None when MA gave no usable musical time -- callers must
    treat that as 'unknown', never guess."""
    item = state.get("current_item") or {}
    dur, elapsed = item.get("duration"), state.get("elapsed_time")
    if dur is None or elapsed is None:
        return None
    try:
        drift = 0.0
        updated = state.get("elapsed_time_last_updated")
        if state.get("state") == "playing" and updated:
            drift = max(0.0, (wall_now if wall_now is not None
                              else time.time()) - float(updated))
        return max(0.0, float(dur) - (float(elapsed) + drift))
    except (TypeError, ValueError):
        return None


def _target_air_at(brk: dict[str, Any] | None) -> datetime | None:
    """The intent's own scheduled target boundary, or None."""
    iso = ((brk or {}).get("telemetry") or {}).get("target_air_at")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def estimated_air_at(state: dict[str, Any],
                     brk: dict[str, Any] | None = None,
                     now: datetime | None = None,
                     wall_now: float | None = None) -> datetime | None:
    """WHEN a break planned against this state would actually be HEARD, or
    None when that cannot be known confidently.

    Spoken text is generated up to RENDER_LEAD_S before the boundary it
    airs on, so render time is NOT airtime: a clock fact computed at render
    is already minutes stale when the listener hears it. This is the one
    place that predicts airtime, from evidence the scheduler already has --
    MA's own musical time plus the intent's existing target -- never a
    fixed offset and never an installation-specific fudge.

    Prediction: now + the current track's remaining musical time, since a
    break airs at the next safe natural boundary -- the end of the item the
    listener is hearing right now.

    An intent's own target_air_at wins only when it lands BEYOND this
    track's boundary window, i.e. more than BOUNDARY_SLACK_S past the
    current track's end (Flow already took the immediate boundary, so the
    break genuinely waits for a later track). Inside that window the target
    must NOT pull the estimate forward: _render_window_open has already
    ruled this track's end an acceptable boundary for that target, so the
    break airs HERE, up to BOUNDARY_SLACK_S before the target. Clamping to
    the target there predicts a moment the listener has already passed --
    breaks are then routinely estimated late, by up to a minute or more,
    and a spoken "It's 8:40 PM" is heard before 8:40.

    Refuses rather than guesses. Returns None when:
      - the player is not PLAYING. A paused or idle player has no knowable
        airtime: wall-clock passage during a pause is not progress toward
        the boundary, so "paused with two minutes left" must never
        become "airs two minutes from now". (An idle queue can report
        elapsed_time frozen with a checkpoint hours old -- naive math
        would predict airtime a few minutes out.)
      - the structural insertion gate is CLOSED (index_in_buffer has moved
        past current_index). This estimate says "the current track's end",
        and that is only the break's boundary while the player has
        not buffered beyond the audible item -- the same condition
        try_insert uses to decide the boundary is still ours. With the
        buffer advanced, an inserted item lands after a track we did not
        measure, so the current track's end is somebody else's boundary
        and the honest answer is 'unknown'.
      - MA gave no usable musical time, or a nonsensical duration.
      - MA's own elapsed_time checkpoint is older than TICK_LIVE_S, the
        existing liveness bound: past it, drift extrapolation is a guess.
      - the estimate lands further out than AIR_ESTIMATE_MAX_S.

    What it cannot foresee, by nature: a listener stopping or resuming
    after the estimate is made. (An estimate can be exactly right when
    made, and a stop and a later resume that restarts the track then push
    the real airtime minutes later. No projection can predict a future
    transport action, and it never makes a false clock -- the NEXT break
    simply refuses, because a stopped player has no knowable airtime.)

    Read-only and side-effect free: it opens no gate, queues nothing, and
    makes no MA call of its own -- it only reads state the caller already
    holds.
    """
    now_dt = now or service_clock.legacy_now()
    if (state or {}).get("state") != "playing":
        return None
    # This function measures ONE boundary: the end of the audible item. That
    # is the break's boundary only while the structural gate is open
    # (the authoritative Flow-mode rule, shared with try_insert); with the
    # buffer advanced the insert lands past a track we never measured.
    import ma_queue
    if not ma_queue.insertion_gate_open(state):
        return None
    item = state.get("current_item") or {}
    try:
        duration = float(item.get("duration"))
        elapsed = float(state.get("elapsed_time"))
    except (TypeError, ValueError):
        return None
    if duration <= 0 or elapsed < 0:
        return None          # nonsensical musical time is not evidence
    updated = state.get("elapsed_time_last_updated")
    if updated is not None:
        # A checkpoint MA stamped long ago is not evidence about where the
        # track is now, and extrapolating across it is a guess. Absent
        # entirely is a different case: the queue read that produced this
        # state is itself fresh, which is exactly why _remaining_s treats
        # it as zero drift.
        try:
            age = (wall_now if wall_now is not None
                   else time.time()) - float(updated)
        except (TypeError, ValueError):
            return None
        if age < 0 or age > TICK_LIVE_S:
            return None
    remaining = _remaining_s(state, wall_now)
    if remaining is None:
        return None
    # An overrun position (position > duration) bounds to zero through the
    # existing lifecycle semantics in _remaining_s: the boundary is due now,
    # which is still a truthful answer.
    estimate = now_dt + timedelta(seconds=remaining)
    target = _target_air_at(brk)
    if (target is not None
            and target > estimate + timedelta(seconds=BOUNDARY_SLACK_S)):
        # Beyond this track's boundary window: the break really is waiting
        # for a later track, and the scheduled target is authoritative.
        # Inside the window (target <= end + slack) this track's end IS the
        # boundary _render_window_open accepted, so the measured end stands.
        estimate = target
    if estimate < now_dt:
        return None          # never a time in the past
    if (estimate - now_dt).total_seconds() > AIR_ESTIMATE_MAX_S:
        return None
    return estimate


def _render_window_open(brk: dict[str, Any], state: dict[str, Any],
                        now_dt: datetime, at_boundary: bool) -> bool:
    """Should this planned intent render NOW?

    True only when (a) the CURRENT track's natural end is the target
    boundary -- it lands at/after target_air_at - BOUNDARY_SLACK_S -- and
    (b) we are within RENDER_LEAD_S of that end, so the spoken context stays
    fresh. Boundary calls act as a degraded fallback when position ticks are
    not flowing (render at track start: correctly anchored, less fresh).
    This decides WHEN TO RENDER only; it is never the insertion gate."""
    target_iso = (brk.get("telemetry") or {}).get("target_air_at")
    if target_iso:
        try:
            target = datetime.fromisoformat(target_iso)
        except (TypeError, ValueError):
            target = now_dt
    else:
        target = now_dt  # legacy/targetless intent: render at first chance
    remaining = _remaining_s(state)
    if remaining is None:
        # No musical time available. Only a boundary may render, and only
        # once the target time itself has arrived.
        return at_boundary and now_dt >= target - timedelta(
            seconds=BOUNDARY_SLACK_S)
    end_at = now_dt + timedelta(seconds=remaining)
    if end_at < target - timedelta(seconds=BOUNDARY_SLACK_S):
        return False  # this track ends before the target: air on a later one
    if remaining <= RENDER_LEAD_S:
        return True
    return at_boundary and not _ticks_alive()


def _render_lead_ok(remaining: float | None) -> bool:
    """Is there still time to write, voice and insert a break before the
    current track ends? `remaining` is _remaining_s: Music Assistant's own
    musical time, drift-corrected on the host's real clock. The service
    clock is deliberately not an input -- a lab jump moves station time,
    not the song. Unknown remaining time is never enough: nothing proves the
    track is still there to render against."""
    return remaining is not None and remaining >= MIN_RENDER_LEAD_S


def _sanity_retry_refusal(token: str, render_deadline: float | None,
                          synth_s: float, db_path=None) -> str | None:
    """Why a render the speech sanity guard rejected may NOT be synthesized
    once more, or None when it may.

    The retry must provably land before its boundary: another synthesis
    as long as the one just measured, plus the MIN_RENDER_LEAD_S a fresh
    render needs to be levelled, probed and inserted. No measured deadline
    is no evidence, and no evidence is no retry. A row that has left
    'rendering' -- revoked for News, an announcement, a cancel -- is not
    ours to re-voice."""
    if render_deadline is None:
        return "no_deadline"
    left = render_deadline - time.monotonic()
    if left < synth_s + MIN_RENDER_LEAD_S:
        return "lead_short"
    brk = get_break(token, db_path)
    if not brk or brk.get("state") != "rendering":
        return "not_rendering"
    return None


def _split_name(name: str | None) -> tuple[str, str]:
    if name and " - " in name:
        artist, title = name.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", (name or "").strip()


def recent_break_texts(limit: int = 3, max_chars: int = 220,
                       db_path=None) -> list[str]:
    """The spoken text of the last few AIRED breaks (newest first), each
    length-capped, for anti-repetition. Bounded and read-only; drawn from
    the same dj_breaks history the rest of the engine uses -- no new store.
    Any failure yields [] (repetition-awareness is a nicety, never a
    blocker)."""
    try:
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT spoken_text FROM dj_breaks WHERE state = 'completed' "
                "AND spoken_text IS NOT NULL AND spoken_text != '' "
                "ORDER BY rowid DESC LIMIT ?", (max(1, min(int(limit), 5)),)
            ).fetchall()
        finally:
            c.close()
        return [(r[0] or "")[:max_chars] for r in rows]
    except Exception:  # noqa: BLE001
        return []


def recent_template_ids(limit: int = 4, db_path=None) -> list[str]:
    """The deterministic writer's template ids from the last few AIRED
    breaks (newest first), for template rotation.

    The ids already live in each break's `writer` telemetry note, so this
    is one bounded read of the same dj_breaks history `recent_break_texts`
    uses -- no new store and no new column. Rows written by a model
    writer carry no template id and are skipped, which is right: the
    deterministic bank only needs to step around ITS own recent shapes.
    Any failure yields [] (rotation is a nicety, never a blocker)."""
    try:
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT json_extract(telemetry, '$.writer.template_id') "
                "FROM dj_breaks WHERE state = 'completed' "
                "ORDER BY rowid DESC LIMIT ?",
                (max(1, min(int(limit), 8)) * 3,)).fetchall()
        finally:
            c.close()
        out = [str(r[0]) for r in rows if r and r[0]]
        return out[:max(1, min(int(limit), 8))]
    except Exception:  # noqa: BLE001
        return []


def recent_dj_memory(limit: int = smart_dj.MAX_RECENT,
                     max_chars: int = smart_dj.RECENT_TEXT_MAX,
                     max_age_h: float = smart_dj.RECENT_MAX_AGE_H,
                     now: datetime | None = None,
                     db_path=None) -> list[dict[str, Any]]:
    """The Smart DJ planner's short memory: the last few ORDINARY breaks
    that actually aired (newest first), each as a bounded record -- the
    planner kind and topic key when the break was planned, the spoken text
    length-capped -- drawn from the dj_breaks ledger the rest of the engine
    already keeps. No new store. One indexed query with a fixed LIMIT, so
    the cost never grows with the station's history; anything older than
    `max_age_h` is not 'recent' and is dropped. Any failure yields [] --
    repetition-awareness is a nicety, never a blocker."""
    now_dt = now or service_clock.legacy_now()
    limit = max(1, min(int(limit), smart_dj.MAX_RECENT))
    out: list[dict[str, Any]] = []
    try:
        c = _connect(db_path)
        try:
            # liners and news intros are 'completed' rows too; scan a
            # bounded handful past the limit so they cannot hide the
            # ordinary breaks, never the whole table
            rows = c.execute(
                "SELECT spoken_text, telemetry, updated_at FROM dj_breaks "
                "WHERE state = 'completed' ORDER BY rowid DESC LIMIT ?",
                (limit * 3,)).fetchall()
        finally:
            c.close()
        cutoff = service_clock.cutoff_destructive(
            now_dt, timedelta(hours=max_age_h))
        for spoken, tel_raw, updated in rows:
            try:
                tel = json.loads(tel_raw or "{}")
            except (TypeError, ValueError):
                tel = {}
            if not is_ordinary_break(tel):
                continue
            stamp = tel.get("air_start") or updated or ""
            if stamp and stamp < cutoff:
                break                       # newest-first: the rest is older
            planner = tel.get("planner") or {}
            # Clipped generously here and FINALLY in smart_dj.build_packet,
            # after its service readings are scrubbed (scrub_recent_text):
            # clipped first, a 100-character memory was mostly the clock.
            entry: dict[str, Any] = {
                "kind": planner.get("action") or "legacy",
                "topic": planner.get("topic_key"),
                "text": (spoken or "")[:max_chars * 4]}
            out.append(entry)
            if len(out) >= limit:
                break
    except Exception:  # noqa: BLE001
        return []
    return out


def _planner_note(decision: dict[str, Any]) -> dict[str, Any]:
    """The bounded telemetry record of one planner decision."""
    return {"action": decision.get("action"),
            "topic_key": decision.get("topic_key"),
            "fact_ids": list(decision.get("fact_ids") or [])[:smart_dj.MAX_FACTS],
            "packet_bytes": decision.get("packet_bytes"),
            "candidates": list(decision.get("candidates") or [])[:len(smart_dj.KINDS)]}


def _planner_for(deps: dict[str, Any] | None):
    """Which Smart DJ planner call, if any, this pass uses.

        deps['planner_fn'] present    -> that (None means: legacy path)
        deps['llm_fn'] injected       -> legacy free-text path (the injected
                                         writer IS the writer, as always)
        DJ_SMART_PLANNER=0            -> legacy path (operational kill switch)
        no LLM writer configured      -> legacy path (deterministic primary:
                                         an install without a model keeps
                                         its breaks, unchanged)
        DJ_SMART_WRITER=p2 and the    -> the P2 personality-first writer
        writer model is the declared     (smart_dj.call_model_p2); any other
        DJ_SMART_WRITER_P2_MODEL         model, or none declared, keeps the
                                         production planner
        otherwise                     -> the production planner call
    """
    deps = deps or {}
    if "planner_fn" in deps:
        return deps["planner_fn"]
    if deps.get("llm_fn") is not None:
        return None
    if os.environ.get("DJ_SMART_PLANNER", "1").strip().lower() in (
            "0", "false", "off", "no"):
        return None
    if not dj_writer.llm_in_use(None):
        return None
    if smart_writer_style() == smart_dj.WRITER_P2:
        try:
            model = smart_dj.p2_configured_model()
        except Exception:  # noqa: BLE001 - resolution fault: production writer
            model = None
        qualified = os.environ.get("DJ_SMART_WRITER_P2_MODEL", "")
        if smart_dj.p2_model_ok(model, qualified):
            return smart_dj.call_model_p2
        _warn_p2_model(model, qualified)
    return smart_dj.call_model


#: DJ_SMART_WRITER: which Smart DJ writer scheduled breaks use. Unset (or
#: anything but "p2") is the production structured planner. "p2" opts in to
#: the P2 personality-first writer (smart_dj.call_model_p2). Read per pass,
#: so a restart is the only way it changes on a running station.
def smart_writer_style() -> str:
    raw = os.environ.get("DJ_SMART_WRITER", "").strip().lower()
    return smart_dj.WRITER_P2 if raw == smart_dj.WRITER_P2 else "smart"


_P2_MODEL_WARNED: set[str] = set()


def _warn_p2_model(model: str | None, qualified: str | None) -> None:
    """Once per model: P2 was asked for but the writer model is not one it
    was declared qualified on (DJ_SMART_WRITER_P2_MODEL), so the production
    planner keeps the slot."""
    key = f"{model}|{qualified}"
    if key not in _P2_MODEL_WARNED:
        _P2_MODEL_WARNED.add(key)
        logger.warning("DJ_SMART_WRITER=p2 ignored: writer model %r is not "
                       "a declared P2 model (DJ_SMART_WRITER_P2_MODEL=%r); "
                       "the production planner writes", model or None,
                       qualified or None)


def _boundary_key(state: dict[str, Any], context: dict[str, Any]) -> str:
    """The identity of the boundary a planner decision belongs to: the MA
    queue_item_id of the track the break would follow (the same identity
    boundary_already_served keys on), or its name when MA gave no id."""
    cur = state.get("current_item") or {}
    if cur.get("queue_item_id"):
        return str(cur["queue_item_id"])
    return "name:" + str(context.get("pre_name") or cur.get("name") or "")


def _planner_declined_here(brk: dict[str, Any], boundary_key: str) -> bool:
    """Did the planner already decide SAY_NOTHING for this intent at THIS
    boundary? One eligible boundary is one planner decision, silence
    included: subsequent observer ticks at the same boundary (and a queue
    rewind that reaches it again) make no second call. The intent simply
    waits for the next boundary, exactly as boundary_replayed does."""
    declined = (brk.get("telemetry") or {}).get("planner_declined") or {}
    return bool(boundary_key) and declined.get("boundary") == boundary_key


#: P2 RETRY BUDGET (smart_dj._decide_p2). A refused P2 draft
#: may get one more generation only inside this budget, measured from the
#: planner stage: the track time left before the boundary this break is for
#: (next_decision_at, from Music Assistant's position), minus the lead a
#: break still needs once its words exist (MIN_RENDER_LEAD_S: TTS, programme
#: audio, the insert and its margin), minus a safety margin that also covers
#: the enrichment this pass already spent -- and never longer than the
#: planner claim, so the claim cannot expire under a retry and let a second
#: pass call the model for the same boundary. No deadline known: no retry.
P2_RETRY_SAFETY_S = float(os.environ.get("DJ_P2_RETRY_SAFETY_S", "5"))


def _p2_retry_budget(now_dt: datetime,
                     next_decision_at: datetime | None) -> float | None:
    if next_decision_at is None:
        return None
    left = ((next_decision_at - now_dt).total_seconds()
            - MIN_RENDER_LEAD_S - P2_RETRY_SAFETY_S)
    return max(0.0, min(left, PLANNER_CLAIM_TTL_S - P2_RETRY_SAFETY_S))


def _planner_claim(token: str, now_dt: datetime, db_path=None) -> bool:
    """Atomically claim the planner decision for a planned intent: a
    conditional UPDATE that stamps planner_started unless a fresh stamp
    (younger than PLANNER_CLAIM_TTL_S) is already there. Exactly one of any
    number of concurrent passes wins; the rest report planner_busy and do
    nothing. Restart-safe because it is the row, and self-healing because
    the stamp expires."""
    stale = service_clock.cutoff_destructive(
        now_dt, timedelta(seconds=PLANNER_CLAIM_TTL_S))
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE dj_breaks SET telemetry = json_set(COALESCE(telemetry, "
            "'{}'), '$.planner_started', ?) WHERE token = ? AND state = "
            "'planned' AND (json_extract(telemetry, '$.planner_started') IS "
            "NULL OR json_extract(telemetry, '$.planner_started') < ?)",
            (_iso(now_dt), token, stale))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def _note_planner_decline(token: str, boundary_key: str,
                          decision: dict[str, Any], now_dt: datetime,
                          db_path=None,
                          next_decision_at: datetime | None = None) -> None:
    """Record SAY_NOTHING for this boundary on the planned row: the decline
    note (bounded), the claim released, and updated_at touched -- the
    intent is alive and deciding, not stale. One statement, one write.
    Best-effort like every telemetry write; a failure here costs at worst
    one extra planner call at the next tick."""
    prior = (get_break(token, db_path) or {}).get("telemetry") or {}
    count = int((prior.get("planner_declined") or {}).get("count") or 0)
    note = {"boundary": boundary_key, "at": _iso(now_dt),
            "reason": decision.get("reason"),
            "packet_bytes": decision.get("packet_bytes"),
            "count": count + 1}
    if decision.get("attempted"):
        note["attempted"] = decision["attempted"]
    if decision.get("guard_category"):
        note["guard_category"] = decision["guard_category"]
    if decision.get("claim"):
        note["claim"] = str(decision["claim"])[:40]
    if next_decision_at is not None:
        # read by render_pressure: no fresh decision before this instant
        note["next_decision_not_before"] = _iso(next_decision_at)
    try:
        c = _connect(db_path)
        try:
            c.execute(
                "UPDATE dj_breaks SET telemetry = json_set(json_remove("
                "COALESCE(telemetry, '{}'), '$.planner_started'), "
                "'$.planner_declined', json(?)), updated_at = ? "
                "WHERE token = ? AND state = 'planned'",
                (json.dumps(note), _iso(now_dt), token))
            c.commit()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        logger.debug("planner decline note failed for %s", token[:8],
                     exc_info=True)


def _smart_dj_stage(token: str, queue_id: str, boundary: str,
                    context: dict[str, Any],
                    resolved: list[dict[str, Any]] | None,
                    service_offered: dict[str, Any], planner,
                    now_dt: datetime, db_path=None,
                    next_decision_at: datetime | None = None) -> str | None:
    """Smart DJ: ONE bounded planner decision for this eligible boundary.

    Runs after the deterministic context enrichment and before render_break,
    for SCHEDULED intents only. Builds the small grounded packet from what
    the scheduler already resolved, makes exactly one model call, validates
    the structured answer, and then either

        returns 'planner_declined'  -- SAY_NOTHING (the model's choice, or
                                       any failure): no render, no TTS, no
                                       audio, no token, no insert; the
                                       decline is keyed to this boundary so
                                       the same boundary never re-plans;
        returns 'planner_busy'      -- another pass holds the claim;
        returns None                -- speak: the validated decision rides
                                       context['smart_dj'] into the existing
                                       render/TTS/programme path, which
                                       stays the only transport authority.

    `boundary` is the queue-item identity of the boundary this decision
    belongs to and `next_decision_at` the earliest instant a fresh decision
    can follow a decline (the current track's natural end), both captured
    by the caller from the live queue state. Nothing here touches the
    queue or the lifecycle state."""
    if not _planner_claim(token, now_dt, db_path):
        return "planner_busy"
    try:
        import dj_context
        local_now = dj_context.station_now(now_dt)
    except Exception:  # noqa: BLE001
        local_now = None
    recent = recent_dj_memory(now=now_dt, db_path=db_path)
    weather_sig = (service_offered or {}).get("weather")
    # dj.atmosphere: licensed optional colour joins the packet, and the
    # answer's sky/light words are screened against the same facts
    colour_facts = _planner_colour(token, queue_id, context, now_dt, db_path)
    packet = smart_dj.build_packet(context, resolved, recent, local_now,
                                   weather_sig=weather_sig)
    size = smart_dj.packet_bytes(packet)
    flags = _authority_flags(context, resolved)
    budget = int(context.get("max_words") or MAX_WORDS)

    try:
        import required_hosting
        context["obligations_due"] = list(required_hosting.obligations(context, budget) or [])
    except Exception:
        pass
    try:
        import dj_context
        if "calendar_evaluation" not in context:
            context["calendar_evaluation"] = dj_context.calendar_evaluation(
                settings.load().get("dj"), now=now_dt)
        if "weather_evaluation" not in context:
            context["weather_evaluation"] = dj_context.weather_evaluation(
                settings.load().get("dj"), now=now_dt, context=context, budget=budget)
    except Exception:
        pass

    def _guard(text: str):
        category = None
        if flags:
            import writer_guard
            category = writer_guard.check(text, flags)
        if category is None and colour_facts is not None:
            import atmosphere
            if atmosphere.screen_spoken(text, colour_facts,
                                        smart_dj._supplied_strings(packet)):
                category = "atmosphere_unlicensed"
        return category

    def _spoken(text: str) -> set[str]:
        import dj_context as _dc
        return _dc.spoken_service_concepts(text, service_offered or {})

    def _warn(text: str) -> list[str]:
        # OBSERVED, NEVER ENFORCED: runs only on an accepted break and
        # cannot change the outcome (writer_guard.warnings: patterns
        # worth counting, but not worth blocking a break on).
        if not flags:
            return []
        import writer_guard
        return writer_guard.warnings(text, flags)

    p2_kwargs: dict[str, Any] = {}
    if getattr(planner, "writer_style", None) == smart_dj.WRITER_P2:
        try:
            p2_prompt = smart_dj.render_p2_prompt(context, packet, local_now,
                                                  budget)
        except Exception:  # noqa: BLE001 - no prompt: refused -> the floor
            logger.warning("p2 prompt render failed", exc_info=True)
            p2_prompt = None
        p2_kwargs = {"p2_prompt": p2_prompt, "p2_context": context,
                     "p2_retry_budget_s": _p2_retry_budget(
                         now_dt, next_decision_at)}
    decision = smart_dj.decide(
        packet, planner,
        clean_fn=lambda t: _clean_spoken(t, max_words=budget),
        guard_fn=_guard, spoken_fn=_spoken, warn_fn=_warn, **p2_kwargs)
    decision["packet_bytes"] = size
    if decision.get("writer_style"):
        gen = decision.get("generation") or {}
        retry = decision.get("retry") or {}
        logger.info(
            "dj_writer style=%s break=%s action=%s reason=%s done=%s "
            "eval_tokens=%s wall_ms=%s attempts=%s retry=%s",
            decision["writer_style"], token[:8],
            decision["action"], decision.get("reason") or "-",
            gen.get("done_reason") or "-", gen.get("eval_count"),
            gen.get("wall_ms"), len(decision.get("attempts") or []) or 1,
            ("ran" if retry.get("attempted") else retry.get("skipped") or "-"))
    decision["candidates"] = list(packet.get("candidates") or [])
    logger.info(
        "dj_planner queue=%s break=%s boundary=%s action=%s topic=%s "
        "facts=%s packet_bytes=%d reason=%s", queue_id, token[:8],
        boundary[:24], decision["action"], decision.get("topic_key"),
        ",".join(decision.get("fact_ids") or []) or "-", size,
        decision.get("reason") or "-")
    # Forensic capture (diagnostic only, fenced, append-only). Written
    # for EVERY decision including silence, because the decline note
    # below keeps only the most recent reason per boundary and its row is
    # reaped within hours, long before anyone reviews it.
    try:
        import dj_forensics
        dj_forensics.decision(token, queue_id, boundary, decision, context,
                              now_dt)
    except Exception:  # noqa: BLE001
        logger.debug("forensic decision capture failed", exc_info=True)
    if decision["action"] == smart_dj.SAY_NOTHING:
        # REQUIRED HOSTING SURVIVES AN ENRICHMENT FAILURE.
        # The model is authoritative over what is worth ADDING, never over
        # whether the station hosts the transition it owes. When the
        # decline is enrichment-level and there is authoritative material
        # to host from, the break falls through to render_break WITHOUT a
        # planner decision, where the deterministic floor writes it. No
        # second model call happens on that path -- required_hosting_only
        # short-circuits the free-text writer too.
        #
        # The decline is still recorded in forensics above (the
        # planner_decision record was written before this branch), so the
        # model's failure stays inspectable after a competent break airs.
        import required_hosting
        reason = decision.get("reason")
        # A riding announcement is owed whatever the model decided: any
        # decline hands the slot to the floor, which reads it verbatim.
        if ((reason in required_hosting.ENRICHMENT_DECLINES
             or required_hosting.announcement_message(context))
                and required_hosting.hostable(context)):
            context["required_hosting_only"] = True
            context["planner_declined_reason"] = reason
            owed = required_hosting.obligations(
                context, context.get("max_words"))
            logger.info(
                "dj break %s: planner declined (%s); required hosting "
                "takes the slot (owes %s)", token[:8], reason,
                ",".join(owed) or "-")
            record = {"planner_reason": reason, "draft_rejected": False,
                      "obligations": owed, "unmet": list(owed),
                      "at": _iso(now_dt)}
            _note(token, {"required_hosting": record}, db_path)
            context["required_hosting_record"] = record
            return None
        _note_planner_decline(token, boundary, decision, now_dt, db_path,
                              next_decision_at=next_decision_at)
        return "planner_declined"
    context["smart_dj"] = decision
    return None


# --- music-aware stories (story_job / dj_memory / split_hosting) ---------------

def _story_history(queue_id: str, token: str | None = None, db_path=None,
                   limit: int = 12) -> dict[str, Any]:
    """What this queue's recent ORDINARY breaks did with stories, from break
    telemetry, newest first and never counting `token` (the break being
    planned, whose row already exists): the act the previous break talked
    about, how many breaks since a full story, and the framing lines and
    template ids aired lately. A story the render guard refused did not
    air and counts as no story."""
    out: dict[str, Any] = {"last_act": "", "breaks_since_story": 99,
                           "framing": [], "template_ids": []}
    try:
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT token, telemetry FROM dj_breaks WHERE queue_id = ? "
                "AND state NOT IN ('revoked', 'failed', 'missed') "
                "ORDER BY created_at DESC LIMIT ?",
                (queue_id, limit + 1)).fetchall()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return out
    seen = 0
    for tok, tel_json in rows:
        if tok == token:
            continue
        try:
            tel = json.loads(tel_json or "{}")
        except (TypeError, ValueError):
            tel = {}
        if not is_ordinary_break(tel):
            continue
        sb = tel.get("story_break") or {}
        if sb.get("rejected"):
            sb = {}
        if seen == 0:
            out["last_act"] = sb.get("artist") or ""
        if sb.get("kind") == "story" and out["breaks_since_story"] == 99:
            out["breaks_since_story"] = seen
        out["framing"] += [p for p in (sb.get("framing") or []) if p]
        out["template_ids"] += list(sb.get("template_ids") or [])
        seen += 1
    return out


def _music_story_stage(token: str, queue_id: str, boundary: str,
                       context: dict[str, Any], resolved, now_dt: datetime,
                       deps: dict[str, Any] | None, db_path=None
                       ) -> dict[str, Any] | None:
    """The music story for THIS break, or None (the existing path hosts it,
    exactly as before). Live only when the story path is active -- the
    listener's preference AND the rollout gate (settings) -- and only
    for an ordinary scheduled break that owes nothing the story path cannot
    carry (no riding announcement, no calendar event). Never raises."""
    deps = deps or DEFAULT_DEPS or {}
    try:
        import story_job
        if not deps.get("stories_active_fn", story_job.active)():
            return None
        if (context or {}).get("purpose") not in (None, "", PURPOSE_SCHEDULED):
            return None
        import required_hosting
        if required_hosting.announcement_message(context) or \
                context.get("calendar_event"):
            return None
        memory = deps.get("story_memory") or story_job.open_memory()
        try:
            return _music_story_hosting(token, queue_id, boundary, context,
                                        resolved, now_dt, memory, db_path)
        finally:
            if memory is not deps.get("story_memory"):
                memory.close()
    except Exception:  # noqa: BLE001 - a story never costs a break
        logger.warning("music story stage failed (existing path hosts)",
                       exc_info=True)
        return None


def _music_story_hosting(token: str, queue_id: str, boundary: str,
                         context: dict[str, Any], resolved,
                         now_dt: datetime, memory, db_path=None
                         ) -> dict[str, Any] | None:
    """_music_story_stage's body: choose, assemble, reserve, record."""
    import split_hosting
    _reconcile_story_selections(memory, token, now_dt, db_path)
    hist = _story_history(queue_id, token, db_path)
    now_s = _story_epoch(now_dt)
    recent = []
    for row in list(resolved or [])[-12:]:
        if isinstance(row, dict) and row.get("outcome") == "played" and \
                row.get("artist"):
            at = _parse_iso(row.get("started_at") or row.get("resolved_at"))
            if at is not None:
                recent.append((row["artist"], at.timestamp()))
    try:
        # the one authoritative on-air name (its own default when none is
        # set); without this import every story break would fall back to
        # saying "the station"
        import settings as _settings
        station = _settings.resolved_station_name() or ""
    except Exception:  # noqa: BLE001
        station = ""
    asides, atmosphere_facts = _atmosphere_asides(token, queue_id, context,
                                                  now_dt, db_path)
    out = split_hosting.host_with_story(
        context, memory, station_name=station or "the station",
        break_token=token, recent_artists=recent,
        breaks_since_story=hist["breaks_since_story"],
        last_break_act=hist["last_act"], now=now_s,
        recent_template_ids=(list(context.get("recent_template_ids") or [])
                             + hist["template_ids"])[-8:],
        recent_framing=hist["framing"][-8:],
        salt=int(now_s // 60), wpm=_story_wpm(),   # budget: spoken seconds
        asides=asides, atmosphere_facts=atmosphere_facts)
    if out is None:
        return None
    mat = out.get("material") or {}
    framing = [p["text"] for p in out["parts"]
               if p.get("role") in ("setup", "payoff")]
    record = {"story_id": mat.get("story_id"), "artist": mat.get("artist"),
              "kind": mat.get("kind"), "about": mat.get("about"),
              "source": {"provider": mat.get("source_provider"),
                         "field": mat.get("source_field"),
                         "ref": mat.get("source_ref")},
              "selection": out.get("selection"),
              "template_ids": out["template_ids"], "framing": framing,
              "obligations_owed": out["obligations_owed"],
              "obligations_met": out["obligations_met"],
              "obligations_unmet": out["obligations_unmet"],
              "spoken_s": out["spoken_s"], "words": out["words"],
              "parts": [{"role": p["role"], "source": p["source"]}
                        for p in out["parts"]],
              "selected_at": _iso(now_dt)}
    if out.get("atmosphere"):
        record["atmosphere"] = out["atmosphere"]
    _note(token, {"story_break": record}, db_path)
    try:
        import dj_forensics
        dj_forensics.decision(token, queue_id, boundary, {
            "action": "story", "text": out["text"], "reason": None,
            "story_id": record["story_id"], "kind": record["kind"],
            "fact_ids": []}, context, now_dt)
    except Exception:  # noqa: BLE001
        logger.debug("forensic story capture failed", exc_info=True)
    logger.info("dj break %s: music story %s (%s about %s, %.1f s)",
                token[:8], record["story_id"], record["kind"],
                record["artist"], out["spoken_s"])
    return {**record, "text": out["text"]}


def _atmosphere_asides(token: str, queue_id: str, context: dict[str, Any],
                       now_dt: datetime, db_path=None
                       ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Atmosphere candidates for this story break, best first, with the facts
    that license them -- ([], {}) unless dj.atmosphere is on. The facts are
    resolved for the moment these words air; the ledger spans every queue."""
    try:
        import settings as _settings
        if not _settings.dj_atmosphere():
            return [], {}
        import atmosphere
        import required_hosting
        air = _parse_iso(context.get("estimated_air_at"))
        facts = atmosphere.facts_now(now_dt, air)
        if not facts:
            return [], {}
        owed = set(required_hosting.obligations(context))
        ledger = atmosphere.ledger_from(_atmosphere_rows(now_dt, db_path))
        return atmosphere.choose(facts, owed=owed, offered=context, ledger=ledger,
                                 queue_id=queue_id, salt=token, at=now_dt), facts
    except Exception:  # noqa: BLE001 - colour never costs a break
        logger.warning("atmosphere candidates failed (story without it)",
                       exc_info=True)
        return [], {}


def _planner_colour(token: str, queue_id: str, context: dict[str, Any],
                    now_dt: datetime, db_path=None) -> dict[str, Any] | None:
    """dj.atmosphere on a Smart DJ break: offer up to two licensed colour
    items (atmosphere.planner_colour) in the context the packet is built
    from, and return the facts the answer is screened against -- {} when no
    snapshot resolves (then no sky or light word is licensed). None when
    atmosphere is off: nothing changes."""
    try:
        import settings as _settings
        if not _settings.dj_atmosphere():
            return None
        import atmosphere
        import required_hosting
        air = _parse_iso(context.get("estimated_air_at"))
        facts = atmosphere.facts_now(now_dt, air)
        if not facts:
            return {}
        owed = set(required_hosting.obligations(context))
        ledger = atmosphere.ledger_from(_atmosphere_rows(now_dt, db_path))
        items = atmosphere.planner_colour(facts, owed=owed, offered=context,
                                          ledger=ledger, queue_id=queue_id,
                                          salt=token, at=now_dt)
        if items:
            context["atmosphere_colour"] = items
            context["atmosphere_colour_text"] = " ".join(i["say"] for i in items)
            context["atmosphere_subjects"] = atmosphere.colour_subjects(items)
            context["atmosphere_colour_facts"] = {
                k: facts.get(k) for k in ("sky", "phase", "weekday")}
        return facts
    except Exception:  # noqa: BLE001 - colour never costs a break
        logger.warning("planner colour failed (break without it)", exc_info=True)
        return None


def _note_planner_colour(token: str, context: dict[str, Any], spoken: str,
                         now_dt: datetime, db_path=None) -> None:
    """The offered colour the accepted words actually carry, recorded with
    the sentence that said it -- the atmosphere ledger's evidence for a
    Smart DJ break. Nothing when the writer left the colour out."""
    try:
        import atmosphere
        names = []
        for key in ("pre_name", "post_name"):
            names += [x.strip() for x in str(context.get(key) or "").split(" - ") if x.strip()]
        used = atmosphere.colour_used(spoken, context["atmosphere_colour"],
                                      context.get("atmosphere_colour_facts") or {},
                                      masks=names)
        if used:
            rec = {k: used.get(k) for k in ("id", "family", "topic", "weather_talk",
                                            "records")}
            rec.update(text=used["said"], sky=(context.get("atmosphere_colour_facts")
                                               or {}).get("sky") if used.get("weather_talk")
                       else None,
                       phase=(context.get("atmosphere_colour_facts") or {}).get("phase"),
                       variant=None, path="planner", aired=True, at=_iso(now_dt))
            _note(token, {"atmosphere": rec}, db_path)
    except Exception:  # noqa: BLE001
        logger.debug("planner colour record failed", exc_info=True)


def _atmosphere_rows(now_dt: datetime, db_path=None) -> list[dict[str, Any]]:
    """Recent breaks on EVERY queue that aired or will (not failed, revoked
    or missed), for atmosphere.ledger_from: a restart or another output
    forgets nothing. Time base: the break's estimated air time, else its
    creation, on the engine's own clock."""
    import atmosphere
    since = _iso(now_dt - timedelta(hours=atmosphere.LEDGER_HOURS))
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT queue_id, created_at, telemetry FROM dj_breaks WHERE "
            "created_at >= ? AND state NOT IN ('failed', 'revoked', 'missed')",
            (since,)).fetchall()
    finally:
        c.close()
    out = []
    for queue, created, raw in rows:
        try:
            tel = json.loads(raw or "{}")
        except (TypeError, ValueError):
            continue
        at = _parse_iso(tel.get("estimated_air_at")) or _parse_iso(created)
        if at is not None and (at.tzinfo is None) == (now_dt.tzinfo is None):
            out.append({"at": at, "queue_id": queue, "telemetry": tel})
    return out


def _note_atmosphere(token: str, context: dict[str, Any] | None, spoken: str,
                     story_accepted: str | None, now_dt: datetime,
                     db_path=None) -> None:
    """What the listener will hear of the story break's aside: its record,
    and whether the accepted words carry it (`aired`). A story the guard
    refused airs the floor, and its aside with it never airs."""
    sb = ((context or {}).get("story_break") or {}).get("atmosphere")
    if not sb:
        return
    aired = bool(story_accepted) and bool(sb.get("text")) and \
        sb["text"] in str(spoken or "")
    _note(token, {"atmosphere": {**sb, "aired": aired,
                                 "at": _iso(now_dt)}}, db_path)


def _reconcile_story_selections(memory, token: str, now_dt: datetime | None,
                                db_path=None) -> None:
    """Settle pending selections against the break rows, the authority. A
    few revoke paths move many rows in one UPDATE (not through _transition),
    and a restart can land between a terminal move and its memory write: a
    pending selection whose break is already terminal is resolved here --
    `completed` confirms, anything else releases. This break's own earlier
    selection (a re-plan of the same token) is released so it can choose
    again."""
    ts = _story_epoch(now_dt)
    pending = memory.pending_tokens()
    if not pending:
        return
    c = _connect(db_path)
    try:
        marks = ",".join("?" for _ in pending)
        rows = {r[0]: (r[1], r[2]) for r in c.execute(
            f"SELECT token, state, telemetry FROM dj_breaks WHERE token IN "
            f"({marks})", pending).fetchall()}
    finally:
        c.close()
    for tok in pending:
        state, tel_json = rows.get(tok, (None, None))
        try:
            tel = json.loads(tel_json or "{}")
        except (TypeError, ValueError):
            tel = {}
        if tok == token:
            memory.release(tok, "replanned", now=ts)
        elif state == STATE_COMPLETED and _story_spoken(tel) and not _not_all_heard(tel):
            memory.confirm_aired(tok, now=ts, evidence="completed")
        elif state == STATE_COMPLETED and _not_all_heard(tel):
            memory.release(tok, "completed_not_all_heard:" + _not_all_heard(tel), now=ts)
        elif state is None or state in TERMINAL_STATES:
            memory.release(tok, state or "unknown_break", now=ts)


def _not_all_heard(tel: dict[str, Any]) -> str | None:
    """The break's `coverage` verdict, when it has one: cut short, or
    interrupted and not resumed where it was (ANNOUNCE NOW). Such a story
    was not heard whole, so it is released -- free to air again -- rather
    than confirmed. No verdict (always the case without that machinery)
    changes nothing."""
    verdict = (tel.get("coverage") or {}).get("verdict")
    return str(verdict) if verdict else None


def _story_spoken(tel: dict[str, Any]) -> bool:
    """Did this break's words come from its story (not the floor)?"""
    sb = tel.get("story_break") or {}
    return bool(sb.get("story_id")) and not sb.get("rejected") and (
        (tel.get("writer") or {}).get("mode") == dj_writer.MODE_STORY_PREPARED)


def _story_wpm() -> float:
    """Speaking rate of the engine the next render speaks with
    (split_hosting.ENGINE_WPM): the router's own decision for the voice in
    force, exactly as default_tts makes it. It must not read a settings
    key directly: a key that does not exist silently budgets every story
    at DEFAULT_WPM."""
    import split_hosting
    try:
        engine = tts_router.decide(dict(dj_voice().get("refs") or {}),
                                   resolve_tts_voice()).get("provider")
        return split_hosting.engine_wpm(engine)
    except Exception:  # noqa: BLE001
        return split_hosting.DEFAULT_WPM


@contextlib.contextmanager
def _story_memory():
    """The DJ memory for one lifecycle write: the injected one (tests), or
    the installation's file, closed after use."""
    injected = (DEFAULT_DEPS or {}).get("story_memory")
    if injected is not None:
        yield injected
        return
    import story_job
    memory = story_job.open_memory()
    try:
        yield memory
    finally:
        memory.close()


def _story_lifecycle(token: str, to_state: str, now: datetime,
                     db_path=None) -> None:
    """The memory follows the break row, the authority: `completed` (airing
    -> completed at the next real music track) is the ONLY confirmed airing;
    revoked / failed / missed release the selection. Fenced; a break with no
    story is one cheap telemetry read."""
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT telemetry FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
        finally:
            c.close()
        tel = json.loads((row[0] if row else None) or "{}")
        sb = tel.get("story_break") or {}
        if not sb.get("story_id") or sb.get("rejected"):
            return
        ts = _story_epoch(now)
        with _story_memory() as memory:
            if to_state == STATE_COMPLETED and _story_spoken(tel) and _not_all_heard(tel):
                memory.release(token, "completed_not_all_heard:" + _not_all_heard(tel), now=ts)
            elif to_state == STATE_COMPLETED and _story_spoken(tel):
                memory.confirm_aired(token, now=ts, evidence="completed")
            else:
                memory.release(token, to_state if to_state != STATE_COMPLETED
                               else "completed_without_story", now=ts)
    except Exception:  # noqa: BLE001
        logger.debug("story lifecycle record failed", exc_info=True)


def _story_rejected(token: str, reason: str | None, now: datetime | None,
                    db_path=None) -> None:
    """The render guard refused the story's words: the floor hosts the
    break, so the selection is released now and the row says why -- a
    completed break must never confirm words that did not air."""
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT telemetry FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
        finally:
            c.close()
        tel = json.loads((row[0] if row else None) or "{}")
        sb = dict(tel.get("story_break") or {})
        if not sb.get("story_id"):
            return
        sb["rejected"] = reason or "rejected"
        _note(token, {"story_break": sb}, db_path)
        with _story_memory() as memory:
            memory.release(token, "rejected",
                           now=_story_epoch(now))
    except Exception:  # noqa: BLE001
        logger.debug("story rejection record failed", exc_info=True)


def _story_epoch(at: datetime | None) -> float:
    """The DJ memory keeps epoch seconds; the instant is the service clock's
    (test_service_clock's lint: no direct clock reads in this module)."""
    import service_clock
    return (at or service_clock.now()).timestamp()


def _parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def _story_prep_pass(queue_id: str, now_dt: datetime,
                     deps: dict[str, Any] | None, db_path) -> None:
    """Prepare stories ahead of air, off the ingestion thread (story_job).
    Inert unless the story path is active; tests opt IN."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            import story_job
            story_job.maybe_prepare(queue_id, now_dt, deps, db_path)
        except Exception:  # noqa: BLE001
            logger.warning("story prep pass failed (nothing affected)",
                           exc_info=True)

    if deps.get("synchronous"):
        if deps.get("story_prep"):
            _run()
        return
    try:
        import story_job
        if not deps.get("stories_active_fn", story_job.active)():
            return                         # no thread at all while inactive
    except Exception:  # noqa: BLE001
        return
    threading.Thread(target=_run, name="story-prep", daemon=True).start()


# How long an unchanged DAYLIGHT line stays withheld after it
# aired, derived from the break cadence (the minimum minutes between audible
# breaks). Time and weather have their own fixed intervals in
# service_presence; this window is daylight's only.
_SERVICE_COOLDOWN_CADENCE_FACTOR = 2.5
_SERVICE_COOLDOWN_FLOOR_S = 900       # >= ~15 min even at a fast cadence
_SERVICE_COOLDOWN_CAP_S = 2700        # never mute a concept beyond 45 min


def _service_cooldown_seconds() -> int:
    secs = cadence_minutes() * 60.0 * _SERVICE_COOLDOWN_CADENCE_FACTOR
    return int(max(_SERVICE_COOLDOWN_FLOOR_S,
                   min(_SERVICE_COOLDOWN_CAP_S, secs)))


def callback_state(now: datetime, session_started_at: str | None = None,
                   db_path=None) -> dict[str, Any]:
    """Cadence facts for the deterministic earlier-session callback, read
    from the existing dj_breaks ledger -- no new table and no durable state
    file. Walks completed breaks newest-first, counting ORDINARY music
    breaks (is_ordinary_break: the same definition cancel targeting and the
    cadence anchor use, so news intros and request liners never count) until
    it reaches the last break that actually aired a callback.

    Any failure yields a state that simply denies the callback: a nicety is
    never worth a break.
    """
    out: dict[str, Any] = {"minutes_since_callback": None,
                           "ordinary_since": 0, "ordinary_total": 0,
                           "session_minutes": 0.0}
    try:
        if session_started_at:
            bounds = service_clock.age_bounds(session_started_at, now)
            if bounds is None:
                raise ValueError("unreadable session_started_at")
            out["session_minutes"] = max(0.0, bounds[1] / 60.0)
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT telemetry, updated_at FROM dj_breaks "
                "WHERE state = 'completed' ORDER BY rowid DESC LIMIT 60"
            ).fetchall()
        finally:
            c.close()
        seen_callback = False
        for tel_json, updated in rows:
            try:
                tel = json.loads(tel_json or "{}")
            except Exception:  # noqa: BLE001
                continue
            if not is_ordinary_break(tel):
                continue          # liners and news intros are not cadence
            if tel.get("earlier_callback"):
                if not seen_callback:
                    stamp = tel.get("air_start") or updated
                    bounds = (service_clock.age_bounds(stamp, now)
                              if stamp else None)
                    if bounds is not None:
                        out["minutes_since_callback"] = max(
                            0.0, bounds[1] / 60.0)
                    seen_callback = True
                out["ordinary_total"] += 1
                continue
            out["ordinary_total"] += 1
            if not seen_callback:
                out["ordinary_since"] += 1
    except Exception:  # noqa: BLE001
        logger.debug("callback cadence state unavailable", exc_info=True)
        return {"minutes_since_callback": 0.0, "ordinary_since": 0,
                "ordinary_total": 0, "session_minutes": 0.0}
    return out


def _enrich_song_context(context: dict[str, Any], session: dict[str, Any],
                         service: dict[str, Any], chattiness,
                         now_dt: datetime, db_path) -> None:
    import song_context
    pre_artist, pre_title = _split_name(context.get("pre_name"))
    pre_path = (session or {}).get("current_path")
    facts = song_context.track_facts(pre_path, pre_artist, pre_title)
    if facts:
        if facts.get("album"):
            context["ending_track_album"] = facts["album"]
        char = song_context.derived_character(facts.get("path"))
        if char:
            context["ending_track_character"] = char
    nxt_char = None
    nfacts = None
    nxt_artist, nxt_title = _split_name(context.get("post_name"))
    if nxt_artist or nxt_title:
        nfacts = song_context.track_facts(None, nxt_artist, nxt_title)
        if nfacts:
            if nfacts.get("album"):
                context["next_track_album"] = nfacts["album"]
            nxt_char = song_context.derived_character(nfacts.get("path"))
            if nxt_char:
                context["next_track_character"] = nxt_char
    # Transition character: deterministic evidence for whether the
    # ending->next move is continuous, a gentle change, or a real contrast.
    # MA's album URI + track number are the preferred album-run/ordering
    # evidence; catalog album name and filename order are the fallback.
    if (context.get("pre_name") or facts) and (nfacts or nxt_artist):
        trans = song_context.classify_transition(
            facts, context.get("ending_track_character"),
            nfacts, context.get("next_track_character"),
            pre_album_uri=context.get("pre_album_uri"),
            post_album_uri=context.get("post_album_uri"),
            pre_track_number=context.get("pre_track_number"),
            post_track_number=context.get("post_track_number"))
        context["transition_character"] = trans.get("character")
        if trans.get("album_run"):
            context["album_run"] = True
        if trans.get("album_sequence"):
            context["album_sequence"] = trans["album_sequence"]
    # Music Knowledge: grounded, scope-gated candidate facts about the
    # music at THIS boundary, from the pack shipped in the image. Bounded
    # (<= music_knowledge.MAX_CANDIDATES), deterministic, offline, and
    # fenced separately -- an unavailable or mismatched pack costs the
    # break its knowledge facts and nothing else. Nothing is marked as
    # used here: retrieval is not airing.
    try:
        import dj_context as _dc
        import fact_relevance
        import music_knowledge
        if music_knowledge.available():
            # RELEVANCE IS DECIDED HERE, not by the writer. The boundary
            # carries what the scheduler already resolved -- the two
            # tracks, their catalog albums, and whether this is an
            # album/artist run -- and the retriever drops any fact with no
            # documented reason to be in THIS transition before the
            # planner ever sees it (fact_relevance).
            try:
                local_now = _dc.station_now(now_dt) or now_dt
            except Exception:  # noqa: BLE001
                local_now = now_dt
            found = music_knowledge.candidates(
                (pre_artist, pre_title, context.get("ending_track_album")),
                ((nxt_artist, nxt_title, context.get("next_track_album"))
                 if (nxt_artist or nxt_title) else None),
                exclude_ids=_dc.recent_fact_ids(now_dt),
                boundary=fact_relevance.boundary(context),
                now=local_now)
            if found.get("facts"):
                context["knowledge_facts"] = found["facts"]
            if found.get("reference"):
                # validator-only grounding (smart_dj hides it from the
                # writer); never marked as aired, never offered
                context["knowledge_reference"] = found["reference"]
            if found.get("diagnostics"):
                context["knowledge_diag"] = found["diagnostics"]
    except Exception:  # noqa: BLE001
        logger.warning("music knowledge unavailable (ordinary break)",
                       exc_info=True)
    context["recent_break_texts"] = recent_break_texts(db_path=db_path)
    context["recent_template_ids"] = recent_template_ids(db_path=db_path)
    opportunity = song_context.semantic_opportunity(
        pre_path, pre_artist, pre_title, nxt_char, service,
        chattiness or "normal", now=now_dt, db_path=db_path)
    if opportunity:
        context["semantic_themes"] = opportunity["themes"]
        context["semantic_tone"] = opportunity["tone"]
        context["semantic_reason"] = opportunity["reason"]
    attach_source_context(context, (session or {}).get("queue_id"),
                          items_fn=None)


def attach_source_context(context: dict[str, Any], queue_id: str | None,
                          *, items_fn: Callable | None = None) -> None:
    """Attach the source-aware music runway and the bounded
    source status to the render context.

    ADDITIVE -- it does not replace the grounded ending/next facts or
    authorize any semantic the source did not supply; program elements are
    excluded from the MUSIC runway. Fenced: source context is a nicety,
    never worth a missed break.

    THE WINDOW MUST STRADDLE PLAYBACK. This read used to be
    `list_queue_items(qid, limit=12)` -- no offset, so items 0-11 -- and
    music_runway then defaulted the current position to 0 when it could not
    find the head among them. On a mature station queue (hundreds of items
    with the head far down it) that described the first twelve items of
    the day as the music around the listener, while reading as an ordinary
    runway. Now the window is anchored on the head BY IDENTITY, and one
    that cannot be anchored raises rather than reporting from the top --
    the fence turns that into no runway at all, which is the honest
    answer."""
    if not queue_id:
        return
    try:
        import ma_queue
        import source_capabilities as _srccap
        qs = ma_queue.queue_state(queue_id)
        context["source_mode"] = _srccap.queue_mode(qs)
        context["source_status"] = _srccap.source_status(qs)
        cur_qiid = (qs.get("current_item") or {}).get("queue_item_id")
        if not cur_qiid:
            return          # no head identity: nothing to anchor a runway on
        read = items_fn or ma_queue.read_window_items
        win = read(queue_id, cur_qiid, current_index=qs.get("current_index"))
        context["music_runway"] = _srccap.music_runway(
            win["items"], cur_qiid, base_position=win["base_position"])
    except Exception:  # noqa: BLE001
        logger.debug("source runway unavailable (ordinary break)",
                     exc_info=True)


#: How far ahead to look for the next real song. A rewind can expose a run
#: of spent programming (dozens of stale program items inside a long
#: future), and the walk must be able to step over a run of them
#: without turning into a queue scan. One small page, read once per
#: render, and a window that finds no music simply means "no next track".
NEXT_MUSIC_WINDOW = 24


def attach_next_items(state: dict[str, Any], queue_id: str,
                      items_fn: Callable | None = None) -> None:
    """Attach a bounded ordered forward window to `state` as `next_items`,
    so next_music_item can walk PAST our own programming to the next song.

    THE WINDOW IS ANCHORED ON THE HEAD BY IDENTITY. `current_index` is a
    positional HINT, never authority: sort_index can have gaps
    and whether Music Assistant's `current_index` is a list position or a
    sort_index is still unproven. This used to read at `current_index + 1`
    -- a window that begins one past the presumed head and therefore can
    never contain it, so a wrong hint could not even be detected and the
    songs one slot off would have been announced as next. Now the head is
    included in the window and located by `queue_item_id`, and the forward
    items are whatever follows THAT position.

    THREE OUTCOMES, deliberately distinct:

        anchored          next_items is the verified forward window
        cannot anchor     next_items = [] -- a positive "we looked and
                          cannot place playback". next_music_item then
                          yields no next track, which is the honest
                          answer: MA's snapshot says the head is X while
                          MA's item list has no X, so nothing derived
                          from that snapshot's adjacency is trustworthy,
                          and its `next_item` least of all.
        cannot read       the key is left ABSENT and next_music_item falls
                          back to filtering MA's single `next_item`. A
                          transport failure teaches us nothing about the
                          queue, so the blinder-but-safe path from the
                          original defence still applies.

    Fenced throughout: grounding is a nicety, never worth a missed break."""
    head_id = (state.get("current_item") or {}).get("queue_item_id")
    if not head_id:
        # No head identity at all -- there is nothing to anchor to, and an
        # adjacency we cannot place is one we will not claim.
        state["next_items"] = []
        return
    # Imported BEFORE the try: the handler below names ma_queue.AnchorNotFound,
    # and an except clause that cannot be evaluated would raise NameError
    # out of the fence instead of costing a little grounding.
    import ma_queue
    try:
        read = items_fn or ma_queue.read_window_items
        win = read(queue_id, head_id, current_index=state.get("current_index"),
                   back=0, ahead=NEXT_MUSIC_WINDOW)
    except ma_queue.AnchorNotFound:
        # We learned something, and it is bad: fail CLOSED.
        logger.debug("next-music window could not be anchored on the head "
                     "(no next track claimed)", exc_info=True)
        state["next_items"] = []
        return
    except Exception:  # noqa: BLE001 - grounding is a nicety, never a break
        logger.debug("next-music window unavailable (using next_item only)",
                     exc_info=True)
        return
    items = win.get("items") if isinstance(win, dict) else None
    if not isinstance(items, list):
        return
    # Everything strictly AFTER the verified head position. back=0 means
    # the head is items[0] on the fast path, but the slow path anchors
    # wherever it found it, so compute rather than assume.
    start = (int(win.get("current_position", 0))
             - int(win.get("base_position", 0)) + 1)
    state["next_items"] = [_brief_of(it) for it in items[max(0, start):]
                           if isinstance(it, dict)]


def _brief_of(item: dict[str, Any]) -> dict[str, Any]:
    """A raw MA queue item reduced to the same shape queue_state reports,
    so next_music_item reads one field spelling whichever source it got.

    `_item_brief` takes the uri from `media_item.uri`, which is where a
    real program element carries it. If it is not there, keep whatever
    uri the item does have rather than dropping it: an unreadable uri
    would make a break look like music, and this defence must fail
    CLOSED."""
    import ma_queue
    brief = ma_queue._item_brief(item) or {}
    if not brief.get("uri"):
        fallback = _uri_of(item)
        if fallback:
            brief["uri"] = fallback
    return brief


def _uri_of(item: dict[str, Any]) -> str:
    """Every spelling Music Assistant uses for an item's uri, in the order
    the rest of the codebase reads them."""
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                 dict) else {}
    return str(media.get("uri") or item.get("uri")
               or item.get("media_uri") or "")


def next_music_item(state: dict[str, Any]) -> dict[str, Any]:
    """The next real MUSIC item after the current one, or {} when there is
    none.

    THE NEXT ITEM IN THE QUEUE IS NOT THE NEXT SONG. Parlorwave inserts its
    own finite programming -- DJ breaks, news bulletins -- into the same
    Music Assistant queue, so the physically adjacent item is routinely a
    break. A queue rewind makes that ordinary: a backward jump puts a run
    of ALREADY-AIRED breaks and bulletins back in the future, and a long
    regression can leave dozens of them ahead of the head.
    Reading MA's `next_item` blindly then hands the scheduler
    `next = "Parlorwave Radio - DJ break"` and the DJ announces its own
    jingle as the next song.

    So walk forward and take the first item that is not ours. Identity is
    media_ref.programming_ref -- the ONE canonical programming check every
    consumer that must keep programming out of music already uses -- which
    is structural and lifecycle-blind ON PURPOSE: a spent break, a fresh
    break, a bulletin and a tombstone are all equally not the next song.

    This is a DEFENCE, not a cleanup: it is correct whether or not stale
    items were ever removed from the queue, so it holds even with hygiene
    off, a tripped latch, or a reconciliation that has not run yet.

    When nothing eligible is in the window the answer is {} -- no next
    track. Saying nothing is always available to the writer, and a
    manufactured next song is a lie on the air."""
    items = state.get("next_items")
    if not isinstance(items, list):
        # No forward window was ATTACHED at all (no queue read happened,
        # or it failed in transport): MA's single `next_item` is all we
        # have. Filter it anyway -- a break filtered down to "no next
        # track" is right, and is exactly the harm this prevents.
        #
        # An EMPTY list is different and is not this case: it means the
        # window was attempted and either held no music or could not be
        # anchored on the head. Both of those are positive answers, and
        # both mean no next track. See attach_next_items.
        nxt = state.get("next_item")
        items = [nxt] if isinstance(nxt, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind, _ = media_ref.programming_ref(_uri_of(item))
        if kind:
            continue                      # ours: furniture, never the song
        return item
    return {}


def _context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Freeze the adjacency this render is grounded on: the track whose end
    the break will follow (pre) and, when MA knows it, the MUSIC that will
    play after the break (post -- the pre-insertion next music item, which
    our NEXT insert pushes one slot later).

    `post` is the next MUSIC item, never merely the next queue item: see
    next_music_item."""
    cur = state.get("current_item") or {}
    nxt = next_music_item(state)
    return {
        "pre_name": cur.get("name"),
        "pre_queue_item_id": cur.get("queue_item_id"),
        "pre_index": state.get("current_index"),
        "post_name": nxt.get("name"),
        "post_queue_item_id": nxt.get("queue_item_id"),
        # Album-run evidence from MA (preferred over catalog strings).
        "pre_album_uri": cur.get("album_uri"),
        "pre_track_number": cur.get("track_number"),
        "post_album_uri": nxt.get("album_uri"),
        "post_track_number": nxt.get("track_number"),
    }


# How far the clock a prepared break SPEAKS may drift from the moment it is
# actually heard. A break is written up to RENDER_LEAD_S before it airs and
# then waits for a natural boundary, so some drift is normal and harmless --
# "it's 7:15" heard at 7:17 is how radio has always sounded. What is not
# harmless is a break that waits out a queue rewind: a break composed at
# 7:15 AM that does not air until nearly 7:45 AM would announce 7:15 to
# the room.
#
# 3:00 is chosen to sit under service_presence.TIME_MIN_GAP_S (4:30, the
# closest two clock announcements may ever land) so this bound can never
# reject a break that the clock cadence would have allowed to speak. A
# stated clock is therefore never more than three minutes wrong.
CLOCK_MAX_DRIFT_S = float(os.environ.get("DJ_CLOCK_MAX_DRIFT_S", "180"))

# "8:40 PM", and the two deterministic equivalent forms dj_context may
# supply ("just after 8 PM", "coming up on 8:45 PM"). The anchor is the
# clock itself in every case; the leading words shift the sense by less
# than the drift bound, so they need no separate arithmetic.
_SPOKEN_CLOCK_RE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?[Mm]\.?")


def _clock_anchor(spoken: str, air_local: datetime) -> datetime | None:
    """The wall-clock instant a spoken clock string names, placed on the day
    of `air_local`. None when the string carries no readable clock.

    A clock has no date, so the reading nearest the projected airtime wins:
    that is what makes "11:58 PM" heard at 00:03 a two-minute drift rather
    than a twenty-four-hour one."""
    m = _SPOKEN_CLOCK_RE.search(spoken or "")
    if not m:
        return None
    hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    if not (1 <= hour <= 12) or minute > 59:
        return None
    hour = (hour % 12) + (12 if mer == "p" else 0)
    try:
        same_day = air_local.replace(hour=hour, minute=minute, second=0,
                                     microsecond=0)
    except ValueError:
        return None
    return min((same_day + timedelta(days=d) for d in (-1, 0, 1)),
               key=lambda dt: abs((dt - air_local).total_seconds()))


def clock_still_true(brk: dict[str, Any],
                     air_local: datetime | None) -> tuple[bool, str]:
    """Would the clock this prepared break SPEAKS still be true on air?

    Only breaks that actually stated a time are examined: `clock_aired` is
    recorded at render from the FINAL text (dj_context.spoken_clock), so a
    break that never mentioned the time makes no claim to invalidate.

    Returns (ok, reason). An unknown station clock is not a failure -- it is
    the one condition under which we cannot tell, and refusing to guess is
    the same rule the planner follows everywhere else. The break then falls
    back on its other guards (adjacency, and the READY TTL that bounds gross
    age), which is strictly what it had before this check existed."""
    spoken = ((brk or {}).get("telemetry") or {}).get("clock_aired")
    if not spoken:
        return True, "no_clock_spoken"
    if air_local is None:
        return True, "station_clock_unknown"
    anchor = _clock_anchor(str(spoken), air_local)
    if anchor is None:
        return True, "clock_unreadable"
    drift = abs((air_local - anchor).total_seconds())
    if drift > CLOCK_MAX_DRIFT_S:
        return False, "stale_clock"
    return True, "clock_fresh"


def _projected_air_local(state: dict[str, Any],
                        now_dt: datetime) -> datetime | None:
    """Station-local wall time at which a break inserted NOW would be heard:
    the current track still has to finish. None when the station clock or
    the remaining time is unknown -- callers must treat that as 'cannot
    tell', never substitute server time."""
    try:
        import dj_context
        local_now = dj_context.station_now(now_dt)
        if local_now is None:
            return None
        remaining = _remaining_s(state)
        return dj_context.station_project(
            local_now, timedelta(seconds=remaining or 0.0))
    except Exception:  # noqa: BLE001 - a nicety, never worth a missed break
        logger.debug("projected airtime unavailable", exc_info=True)
        return None


def pre_air_revalidate(brk: dict[str, Any],
                       state: dict[str, Any],
                       air_local: datetime | None = None
                       ) -> tuple[bool, str]:
    """Are a prepared break's assumptions STILL true, checked immediately
    before it is claimed for air? Returns (ok, reason).

    An audio file having already been generated is not a reason to say
    something untrue. Between render and air the queue can be edited,
    skipped, or rewound by Music
    Assistant restarting a flow stream, which can leave a break waiting
    a long time for a boundary that never stopped being "next".

    Two classes of assumption, each checked against live evidence:

      adjacency  the track this break says it followed, and the one it says
                 is coming, are still those tracks (context_still_valid)
      the clock  a stated time is still the time (clock_still_true)

    Deliberately NOT re-synthesised or patched: a prepared break that is
    wrong is discarded and replanned from live context, never edited after
    the fact. Rewriting the words under finished audio would make the text
    and the sound disagree, which is worse than silence."""
    if not context_still_valid(brk, state):
        return False, "stale_context"
    return clock_still_true(brk, air_local)


def boundary_already_served(queue_id: str, pre_queue_item_id: str | None,
                            exclude_token: str | None = None,
                            db_path=None) -> dict[str, Any] | None:
    """The break Parlorwave ALREADY placed at the boundary after this exact
    queue item, or None (duplicate-boundary containment).

    Identity is the Music Assistant queue_item_id the render was grounded
    on (telemetry intended_pre_queue_item_id, written by render_break for
    every scheduled break) -- never a title, an artist or a queue index.
    Evidence that the break really occupied that boundary is placement:
    the row reached 'queued' (so its audio item went into the queue right
    after that item) and is now queued, airing or completed, or was
    missed/revoked AFTER placement (the enqueue stamp). A break that was
    rendered for this item but revoked before it was ever inserted made no
    use of the boundary and does not count, so the ordinary replan after a
    stale-context revocation is unaffected.

    THE PASS, NOT THE QUEUE'S LIFETIME. A break counts only if
    it was placed in the CURRENT playback pass (begin_pass). Queue item ids
    outlive any one pass through them, so an unscoped check would turn "the
    listener started the finished queue over" into a whole pass of songs
    the DJ refuses to host. What it still stops is what it was written for: the
    same transition reached again inside one pass -- MA's flow-restart
    rewind, a reconnect, a retried delivery.

    Why this exists: Music Assistant's flow-restart resume can rewind the
    ordered queue by a few positions, and each lap
    through the same items reaches the same boundaries again. The cadence
    is a clock, so on a lap where a break happens to be due at a boundary
    the station has already spoken at, the planner would speak there
    again. This guard says: the same boundary is not a new opportunity;
    the intent simply waits for the next boundary, exactly as a closed
    insertion gate makes it wait. Fails OPEN on an unreadable store -- a
    containment guard is never a reason to silence the DJ."""
    if not queue_id or not pre_queue_item_id:
        return None
    try:
        c = _connect(db_path)
        try:
            row = c.execute(
                "SELECT token, state, updated_at FROM dj_breaks "
                "WHERE queue_id = ? AND token != ? "
                "AND json_extract(telemetry, '$.intended_pre_queue_item_id') = ? "
                "AND (state IN ('queued', 'airing', 'completed') "
                "     OR (state IN ('missed', 'revoked') "
                "         AND json_extract(telemetry, '$.enqueue') IS NOT NULL)) "
                "AND COALESCE(json_extract(telemetry, '$.enqueue'), "
                "             updated_at) >= ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (queue_id, exclude_token or "", pre_queue_item_id,
                 pass_started_at(queue_id, db_path) or "")).fetchone()
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - containment guard, never a gag
        logger.warning("boundary-replay check unreadable; not suppressing",
                       exc_info=True)
        return None
    if row is None:
        return None
    return {"token": row[0], "state": row[1], "updated_at": row[2]}


# One log line per replayed boundary, not one per tick while it is current.
_replay_logged: tuple[str, str] | None = None

_PASS_KEY = "dj_pass_started_at:"


def begin_pass(queue_id: str, reason: str, now: datetime | None = None,
               db_path=None) -> str | None:
    """A NEW playback pass through this queue starts now (queue_end: the
    listener started a finished queue over, or MA's Repeat wrapped it).
    Breaks placed before it no longer make a boundary "already served".
    Durable, so a restart mid-pass keeps the pass. Returns the stamp, or
    None when it could not be written (the old, unscoped answer stands)."""
    global _replay_logged
    # microseconds: a break placed earlier in the same second compares as
    # earlier (telemetry stamps are whole seconds), never as this pass's
    stamp = (now or service_clock.legacy_now()).isoformat(timespec="microseconds")
    try:
        c = _connect(db_path)
        try:
            c.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                      (_PASS_KEY + queue_id, stamp))
            c.commit()
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - never a reason to silence the DJ
        logger.warning("playback pass not recorded", exc_info=True)
        return None
    _replay_logged = None
    logger.info("dj_playback_pass_started queue=%s reason=%s at=%s",
                queue_id, reason, stamp)
    return stamp


def pass_started_at(queue_id: str, db_path=None) -> str | None:
    """When the current playback pass through this queue began, or None."""
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT value FROM meta WHERE key = ?",
                            (_PASS_KEY + queue_id,)).fetchone()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None
    return row[0] if row else None


def _attach_successor_window(brk: dict[str, Any] | None,
                             state: dict[str, Any], queue_id: str,
                             items_fn: Callable | None = None) -> None:
    """Before a pre-air check, give `state` the same forward window the
    render walked, so context_still_valid compares next MUSIC with next
    MUSIC.

    Reads nothing unless it could change the answer: the break must have
    made a successor claim, and MA's raw next_item must not already be a
    song (when it is, the walk would stop right there). The window read is
    attach_next_items itself -- same anchoring on the head, same fail-
    closed outcomes -- so there is one definition of "next music"."""
    if not isinstance(state, dict) or "next_items" in state:
        return
    if not ((brk or {}).get("telemetry") or {}).get(
            "intended_post_queue_item_id"):
        return
    nxt = state.get("next_item")
    if isinstance(nxt, dict) and not media_ref.programming_ref(
            _uri_of(nxt))[0]:
        return
    attach_next_items(state, queue_id, items_fn)


def context_still_valid(brk: dict[str, Any], state: dict[str, Any]) -> bool:
    """Is the rendered break's recorded adjacency still true on the live
    queue? BOTH sides of the break are validated:

    - the intended predecessor must still be the CURRENT item (same
      queue_item_id, same index) -- otherwise "that was X" is a lie;
    - when the prompt was given a next_track (intended_post recorded at
      render), the intended successor must still be the NEXT MUSIC item
      (next_music_item) -- otherwise "coming up, B" is a lie even though
      the predecessor never changed.

    The successor is checked as the SAME concept it was rendered from.
    _context_from_state records the next MUSIC item, walking past our own
    programming; comparing that against MA's raw `next_item` asked a
    different question, and a spent break or bulletin sitting between the
    two songs (a rewind leaves runs of them) made every prepared break
    "stale" -- renders revoked one after another, re-planned every 30 s,
    none aired. Our own finite programming never invalidates
    the claim; a different song, a removed song or a new song ahead does.
    Callers attach the forward window first (_attach_successor_window);
    without one, next_music_item filters the raw next_item, which fails
    closed exactly as before.

    A skip, queue edit, or passed boundary breaks either side, and
    knowingly-wrong context must never air. Breaks with no recorded
    adjacency (legacy rows, direct render calls) made no adjacency claim,
    so there is nothing to violate."""
    return context_mismatch(brk, state) is None


def context_mismatch(brk: dict[str, Any], state: dict[str, Any]
                     ) -> tuple[str, str] | None:
    """context_still_valid, reason-coded: None when the adjacency holds,
    else (code, detail) naming WHICH side broke -- `head_changed` (the
    predecessor is no longer the current item: detail `item` or `index`)
    or `successor_mismatch` (the next music is no longer the one the
    break announces). The one definition both the decision and its
    telemetry read."""
    tel = brk.get("telemetry") or {}
    want_id = tel.get("intended_pre_queue_item_id")
    if not want_id:
        return None
    cur = state.get("current_item") or {}
    if cur.get("queue_item_id") != want_id:
        return INSERT_HEAD_CHANGED, "item"
    want_idx = tel.get("intended_pre_index")
    if want_idx is not None and state.get("current_index") != want_idx:
        return INSERT_HEAD_CHANGED, "index"
    want_post = tel.get("intended_post_queue_item_id")
    if want_post:
        if next_music_item(state).get("queue_item_id") != want_post:
            return INSERT_SUCCESSOR_MISMATCH, "next_music"
    return None


def _shift_pass(now_dt: datetime, deps: dict[str, Any]) -> None:
    """DJ Shifts: settle the current shift's DJ voice before anything
    in this pass could render with it (voice_library.reconcile_shift). A
    read and nothing else unless automatic shifts are on and a boundary
    was crossed; it chooses a VOICE and never a break -- no speech, no
    render, no queue. Deliberately ahead of the speech gate: with Speak DJ
    breaks off the selection still follows the shift, so the next break
    after speech returns is in the right voice. Fenced: a shift problem
    must never cost a scheduling pass."""
    try:
        import voice_library
        voice_library.reconcile_shift(now_dt, deps.get("voice_fetch_fn"))
    except Exception:
        logger.warning("DJ shift reconcile failed (the current DJ stays)",
                       exc_info=True)


def schedule_pass(queue_id: str, session: dict[str, Any],
                  now: datetime | None = None,
                  deps: dict[str, Any] | None = None,
                  at_boundary: bool = False, db_path=None) -> str:
    """One airtime-aware scheduling evaluation (boundary hooks and position
    ticks both funnel here). Returns a short outcome string for telemetry
    and tests:

        idle           nothing to do (no intent possible / in-flight break)
        no_state       MA unreadable -- closed world, do nothing
        waiting        intent exists; target boundary/render window not here
        warming        target track playing; LLM warm fired; render soon
        render_failed  render failed (terminal row; cooldown guards retry)
        stale_revoked  the prepared break's assumptions stopped being true
                       -- its adjacency, or the clock it speaks -> revoked
                       and replanned from live context. A break that was
                       already READY and is refused for the clock alone
                       continues ONCE into the replacement when enough
                       real track is left (see
                       _stale_clock_continuation); the pass then returns
                       the replacement's own outcome instead
        render_lead_short  a position tick found a scheduled intent's
                       render window open with less than MIN_RENDER_LEAD_S
                       of real track left: nothing is written (it would be
                       finished after the boundary it is for), and the
                       intent stays planned for the next boundary
        news_in_flight the news event is committed to the queue (queued /
                       intro_airing / bulletin_airing); no new ordinary
                       speech may even be planned behind it
        quiet_guard    a bulletin just aired; ordinary speech waits until a
                       post-news music track is playing, so the first
                       boundary it can take is that track's natural end
        boundary_replayed  the current item already had a Parlorwave break
                       placed after it (a rewound queue reached the same
                       boundary again): the intent is kept and simply
                       waits for the next boundary -- no render, no insert
        planner_declined  the Smart DJ planner decided SAY_NOTHING at this
                       boundary (its own choice, or any planner failure):
                       no render, no audio, no token, no insert; the
                       intent is kept and waits for the NEXT boundary, and
                       this boundary is never re-planned (smart_dj)
        planner_busy   another pass holds the planner claim for this intent
        speech_off_waiting  DJ speech is switched off and the ordinary
                       intent's opportunity has not arrived yet: nothing is
                       written, and the LLM is not warmed
        dj_speech_disabled  DJ speech is switched off and the ordinary
                       opportunity arrived: the intent is closed (revoked,
                       reason dj_speech_disabled) with no render, no TTS
                       and no insert, and the next intent targets one
                       cadence later -- the same skip machinery as Cancel
                       Next Break, so cadence keeps running while silent
        bulletin_due   the next safe boundary is a scheduled bulletin's --
                       the station clock is inside the bulletin's air
                       window, or the boundary this pass would take is
                       predicted to land inside it -- so the
                       cadence break is deferred rather than racing it for
                       the only boundary the bulletin can use
        queued/gate_closed/not_ready/failed  -- try_insert outcomes

    Transport authority is unchanged: try_insert's structural gate decides
    insertion; everything above it only decides when to render and whether
    the rendered context is still true."""
    now_dt = now or service_clock.legacy_now()
    deps = deps or {}
    _shift_pass(now_dt, deps)
    stale_clock: dict[str, Any] = {}
    outcome = _schedule_once(queue_id, session, now_dt, deps, at_boundary,
                             db_path, stale_clock)
    if not stale_clock.get("replacement"):
        return outcome
    return _stale_clock_continuation(queue_id, session, now_dt, deps,
                                     at_boundary, db_path, stale_clock)


def _stale_clock_continuation(queue_id: str, session: dict[str, Any],
                              now_dt: datetime, deps: dict[str, Any],
                              at_boundary: bool, db_path,
                              stale_clock: dict[str, Any]) -> str:
    """LATE-HOSTING RECOVERY: a break that was already READY when this pass
    took it up was refused for its clock alone, revoked, and a replacement
    planned; let that replacement run through the ordinary scheduler rules
    now, in this same pass, rather than on the next one. A break this pass
    rendered itself never hands on (one render per pass).

    Why: the refusal usually happens at a boundary pass -- a READY break is
    offered for insertion the moment a track starts -- and the next pass is
    the first position tick, which the observer schedules 30 s after the
    boundary. On a 35 s track that tick has 0-5 s left, the replacement's
    render overruns the boundary and is revoked head_changed just past the
    next one, and every later tick is late in the same way. The refusal
    itself is unchanged: the stale
    row is revoked stale_pre_air with its stale_clock evidence intact, and
    nothing of its text is reused.

    Bounded twice over. The replacement continues only while the current
    track still has MIN_RENDER_LEAD_S of real time ahead -- measured here on
    a fresh read, not assumed from at_boundary -- otherwise it stays planned
    for the next boundary and this pass ends as it always did. And the
    continuation is one more _schedule_once given no `stale_clock` record,
    so whatever happens inside it, including another stale refusal, can
    never offer a second one. Every ordinary rule applies to the
    replacement exactly as on any pass: News and announcement blocks,
    speech control, the render window, duplicate-boundary containment,
    the planner, the render claim, pre-air revalidation and the structural
    gate."""
    import ma_queue
    read_state = deps.get("state_fn") or ma_queue.queue_state
    try:
        state = read_state(queue_id)
    except Exception:  # noqa: BLE001 - unreadable: the replacement waits
        return "stale_revoked"
    remaining = _remaining_s(state)
    if not _render_lead_ok(remaining):
        logger.info("dj break %s refused for its clock; replacement %s waits "
                    "for the next boundary (%s s of track left, need %.0f)",
                    stale_clock["stale"][:8], stale_clock["replacement"][:8],
                    "unknown" if remaining is None else f"{remaining:.1f}",
                    MIN_RENDER_LEAD_S)
        return "stale_revoked"
    logger.info("dj break %s refused for its clock; replacement %s continues "
                "in this pass (%.1f s of track left)", stale_clock["stale"][:8],
                stale_clock["replacement"][:8], remaining)
    return _schedule_once(queue_id, session, now_dt, deps, at_boundary,
                          db_path, None, seed_state=state)


def _schedule_once(queue_id: str, session: dict[str, Any], now_dt: datetime,
                   deps: dict[str, Any], at_boundary: bool, db_path,
                   stale_clock: dict[str, Any] | None,
                   seed_state: dict[str, Any] | None = None) -> str:
    """One evaluation of schedule_pass's rules (see there for outcomes).

    `stale_clock` receives {stale, replacement} when a READY break is
    refused for its clock and a replacement is planned; None means this
    evaluation may not offer a continuation. `seed_state` is the pass's one
    queue read when the caller already made it, so a continuation decides
    on the same evidence its lead was measured from."""
    import ma_queue
    read_state = deps.get("state_fn") or ma_queue.queue_state
    # ONE queue read per pass, at most. The bulletin deferral below may
    # need to know where the next boundary lands before anything is
    # planned; the same read then serves the render/insert decisions, so
    # asking the question costs no second round-trip to Music Assistant.
    cached: dict[str, Any] = ({} if seed_state is None
                              else {"state": seed_state})

    def _read():
        if "state" not in cached:
            cached["state"] = read_state(queue_id)   # may raise: unreadable
        return cached["state"]

    def _next_boundary_local():
        """Where the next safe boundary lands on the STATION clock, or None
        when it cannot be known -- the same projection the news planner
        makes (estimated_air_at onto station_now), so the DJ and the
        planner are reasoning about one instant. Never raises: an
        unreadable queue is an unknowable boundary, and unknowable means
        the clock rule alone decides (fail open, never silence the DJ)."""
        try:
            st = _read()
            air_at = estimated_air_at(st, None, now_dt)
            if air_at is None:
                return None
            import dj_context
            local_now = dj_context.station_now(now_dt)
            if local_now is None:
                return None
            return dj_context.station_project(local_now, air_at - now_dt)
        except Exception:  # noqa: BLE001
            return None

    def _current_is_music():
        """Is MA's current item real music, on this pass's one read? The
        quiet guard's second piece of evidence: our own bulletin
        or break is never the song a post-news break may follow. Raises
        through to the guard, which treats unreadable as not music."""
        cur = _read().get("current_item")
        return bool(cur) and _is_music_item(cur)

    # A news bulletin that just aired owns the silence after it: ordinary
    # SCHEDULED speech does not stack straight onto the back of a bulletin. The
    # guard releases on observed evidence (one complete music track), not a
    # timer, and it blocks the whole pass so nothing is even rendered to sit
    # ready behind the bulletin -- while the first post-news song plays, only
    # its END may be planned for. Event liners are deliberately unaffected --
    # a listener pressing Start Radio is not the DJ chattering.
    #
    # The bulletin_due deferral is decided against the BOUNDARY this pass
    # would take, not against the clock alone: a break renders
    # RENDER_LEAD_S early and is queued the moment the gate is open, so by
    # the time the air window opened the boundary inside it was already
    # Music Assistant's and nothing could take it back.
    try:
        import news_schedule
        blocked = news_schedule.ordinary_dj_block_reason(
            session, queue_id, db_path=db_path, now=now_dt,
            boundary_fn=_next_boundary_local,
            current_is_music_fn=_current_is_music)
        if blocked:
            _trace_passed_over(queue_id, INSERT_OTHER, str(blocked),
                               cached.get("state"), now_dt, db_path)
            return blocked
    except Exception:
        logger.warning("news block check failed (ignored)", exc_info=True)
    # A committed announcement owns its boundary the same way: ordinary
    # scheduled speech is not rendered to stack behind (or race) it. The
    # check reads only Parlorwave's own durable state and fails open --
    # the structural gate still serializes actual insertion.
    try:
        import announcements
        a_blocked = announcements.ordinary_dj_block_reason(queue_id,
                                                           db_path=db_path)
        if a_blocked:
            _trace_passed_over(queue_id, INSERT_OTHER, str(a_blocked),
                               cached.get("state"), now_dt, db_path)
            return a_blocked
    except Exception:
        logger.warning("announcement block check failed (ignored)",
                       exc_info=True)
    # An ANNOUNCEMENT row is never adopted here: its copy is the operator's
    # and its one door to Music Assistant is announcements._insert_ready,
    # which records durable hand-off evidence before the claim. Adopting it
    # would render DJ copy over it and insert it with no such evidence.
    # The purpose is read in Python, not with json_extract (sqlite raises on
    # a malformed value), and a row whose telemetry cannot be read is
    # skipped rather than adopted: its purpose is unknown, so it could be
    # an announcement.
    c = _connect(db_path)
    try:
        candidates = c.execute(
            "SELECT token, telemetry FROM dj_breaks WHERE queue_id = ? AND "
            "state IN ('planned', 'ready') ORDER BY created_at DESC",
            (queue_id,)).fetchall()
    finally:
        c.close()

    def _purpose_of(raw: Any) -> str | None:
        try:
            tel = json.loads(raw or "{}")
        except (TypeError, ValueError, RecursionError):
            return None
        return break_purpose(tel) if isinstance(tel, dict) else None
    row = next((r for r in candidates
                if _purpose_of(r[1]) not in (None, PURPOSE_ANNOUNCEMENT)),
               None)
    token = row[0] if row else plan_break(queue_id, session, now_dt, db_path)
    if token is None:
        return "idle"
    brk = get_break(token, db_path)
    if brk is None:
        return "idle"
    # A break this pass renders itself is not a break that waited: only one
    # that was already READY may hand its boundary to a replacement below.
    adopted_ready = brk["state"] == "ready"
    try:
        state = _read()
    except Exception as exc:
        _trace_passed_over(queue_id, INSERT_QUEUE_UNREADABLE,
                           f"queue_read:{type(exc).__name__}", None, now_dt,
                           db_path)
        return "no_state"
    if brk["state"] != "ready":
        # a newer intent was chosen over a break that is already READY: that
        # break gets no insertion attempt on this pass
        _trace_passed_over(queue_id, INSERT_OTHER, "shadowed_by_newer_intent",
                           state, now_dt, db_path, exclude=token)
    # DJ SPEECH OFF. Decided here -- after the news/announcement guards and
    # the intent choice, before any warm, context gathering, planner call,
    # render or insert -- and only for ORDINARY DJ speech. Liners and news
    # intros never reach this branch as ordinary, and announcement rows are
    # never adopted above, so neither is touched by it.
    if (is_ordinary_break(brk.get("telemetry"))
            and not dj_speech_enabled()):
        if brk["state"] == "planned" and not _render_window_open(
                brk, state, now_dt, at_boundary):
            return "speech_off_waiting"
        if _close_for_speech(token, queue_id, brk, session, now_dt, db_path):
            return SPEECH_DISABLED
        return "idle"
    # Manual intents are pinned to a specific current item; enforce the pin
    # before any render/insert decision (one bounded retarget, then terminal).
    planner_boundary = ""
    planner_next_at: datetime | None = None
    manual_outcome = _manual_pin_check(brk, state, now_dt, db_path)
    if manual_outcome is not None:
        # Terminated or re-pinned: either way this pass is done -- a
        # retargeted intent renders on a later tick against its NEW
        # boundary, with the normal window rules.
        return manual_outcome
    if brk["state"] == "planned":
        if not _render_window_open(brk, state, now_dt, at_boundary):
            # Warm the LLM once when the target track is already playing
            # (render lands within this track) but the lead window hasn't
            # opened yet.
            remaining = _remaining_s(state)
            target_iso = (brk.get("telemetry") or {}).get("target_air_at")
            if (remaining is not None and target_iso
                    and "llm_warm" not in (brk.get("telemetry") or {})):
                try:
                    target = datetime.fromisoformat(target_iso)
                except (TypeError, ValueError):
                    target = now_dt
                end_at = now_dt + timedelta(seconds=remaining)
                if end_at >= target - timedelta(seconds=BOUNDARY_SLACK_S):
                    _stamp(token, "llm_warm", now_dt, db_path)
                    warm = deps.get("warm_fn", warm_llm)
                    threading.Thread(target=warm, daemon=True).start()
                    return "warming"
            return "waiting"
        if break_purpose(brk.get("telemetry")) == PURPOSE_SCHEDULED:
            # Duplicate-boundary containment: a scheduled break never
            # renders against a predecessor Parlorwave already placed a
            # break after. Manual intents are exempt -- the listener asked
            # for this boundary.
            pre_id = (state.get("current_item") or {}).get("queue_item_id")
            served = boundary_already_served(queue_id, pre_id, token, db_path)
            if served is not None:
                global _replay_logged
                if _replay_logged != (queue_id, pre_id):
                    _replay_logged = (queue_id, pre_id)
                    logger.info(
                        "dj_boundary_replayed queue=%s boundary_item=%s "
                        "prior_break=%s prior_state=%s: no second break at "
                        "a boundary already served (intent kept for the "
                        "next boundary)", queue_id, pre_id,
                        served["token"][:8], served["state"])
                return "boundary_replayed"
            # Captured HERE, from the live queue state, before the
            # enrichment below rebinds `state` to the presence ledger.
            planner_boundary = _boundary_key(state, {})
            if _planner_declined_here(brk, planner_boundary):
                return "planner_declined"
            remaining = _remaining_s(state)
            if remaining is not None:
                planner_next_at = now_dt + timedelta(seconds=remaining)
            if not at_boundary and not _render_lead_ok(remaining):
                # Too late on THIS track: anything written now is finished
                # after the boundary it is for. The intent stays planned and
                # the next boundary renders it with the whole track ahead.
                # Boundary passes are exempt: a track start is the full
                # track this rule protects. (The observer's one reconcile
                # track_start after a restart or target change also arrives
                # as a boundary, possibly mid-track; that predates this rule
                # and costs at most one render, never a run of them.)
                logger.info("dj break %s render held: %.1f s of track left "
                            "on a tick, need %.0f; waiting for the next "
                            "boundary", token[:8], remaining or 0.0,
                            MIN_RENDER_LEAD_S)
                return "render_lead_short"
        # The boundary on the host's monotonic clock, read from the live
        # queue BEFORE the enrichment below rebinds `state`: the only
        # evidence render_break has for whether one re-render still fits.
        # Unknown musical time is no evidence, and gets none.
        lead_s = _remaining_s(state)
        render_deadline = (None if lead_s is None
                           else time.monotonic() + lead_s)
        # Walk past our own programming to the real next song. One
        # bounded page, fenced; see attach_next_items.
        attach_next_items(state, queue_id, deps.get("items_fn"))
        context = _context_from_state(state)
        # Who writes this break decides who owns the calendar. When the
        # Smart DJ planner is the writer, the deterministic topic editor
        # must NOT also select a calendar topic: its `upcoming_event`
        # sentence is written into break_packet's prompt, which the
        # planner path never calls -- while its suppression mark was
        # still committed by every break that rendered. That pairing would
        # keep a correctly published, correctly qualified calendar from
        # ever being heard on air: each eligible event would silently
        # spend its one mention on a break that never mentioned it.
        scheduled = break_purpose(brk.get("telemetry")) == PURPOSE_SCHEDULED
        planner_fn = _planner_for(deps) if scheduled else None
        planner_active = planner_fn is not None
        calendar_marks = None
        # Contextual enrichment (weather/calendar/news): the deterministic
        # topic editor picks eligible facts from the latest HA snapshot,
        # bounded by the break_length composition policy. Purely additive to
        # the prompt -- adjacency validation, the structural gate, and every
        # transport decision are untouched, and any context failure yields a
        # music-only break.
        marks = None
        extras: dict[str, Any] = {}
        service_aired: dict[str, str] = {}
        service_offered: dict[str, Any] = {}
        chat_policy = dj_persona.chattiness_policy(None)
        try:
            import dj_context
            import settings as _settings
            dj_cfg = _settings.load().get("dj") or {}
            profile, chattiness = persona_in_force(dj_cfg)
            chat_policy = dj_persona.chattiness_policy(chattiness)
            context["persona_profile"] = profile["persona_id"]
            context["chattiness"] = chattiness
            st_name = _settings.resolved_station_name()
            context["station_name"] = st_name
            context["station"] = {"name": st_name}
            policy = _length_policy()
            # break_length sets the base ceiling; chattiness scales it and
            # adds the editorial TARGET the writer actually aims for. Only
            # a ceiling was ever stated before, which is why a long+chatty
            # station still got one-sentence breaks: nothing had asked for
            # more (dj_persona.word_budget).
            budget = dj_persona.word_budget(policy["max_words"], chattiness)
            context["max_words"] = budget["max_words"]
            context["target_words"] = (budget["target_low"],
                                       budget["target_high"])
            # Chattiness also bounds the contextual topic budget on top of
            # the break_length policy; it never raises that.
            max_topics = int(policy["max_topics"])
            cap = chat_policy.get("max_topics_cap")
            if cap is not None:
                max_topics = min(max_topics, int(cap))
            allow = policy["topics"]
            if planner_active:
                allow = tuple(t for t in allow if t != "calendar")
                # Calendar awareness: the bounded, fully pre-computed
                # event record for the planner packet. Every number in it
                # -- the band, the minutes, the lead phrase, the clock --
                # is computed here against now_dt; the model is never
                # given a timestamp to subtract. The mark it returns is
                # held back until the aired copy actually names the
                # event (below).
                cal_event, calendar_marks = dj_context.calendar_packet(
                    dj_cfg, now_dt)
                if cal_event:
                    context["calendar_event"] = cal_event
                else:
                    calendar_marks = None
            extras, marks = dj_context.select_topics(
                dj_cfg, now_dt, max_topics=max_topics, allow=allow)
            # Clock facts are stated for the moment the words are HEARD,
            # not the moment they are written: this render runs up to
            # RENDER_LEAD_S before the boundary. No trustworthy estimate
            # means no stated time at all, never a stale one.
            air_at = estimated_air_at(state, brk, now_dt)
            service = dj_context.service_facts(now_dt, at=air_at)
            if not (dj_cfg.get("content") or {}).get("weather", True):
                # dj.content.weather off means no weather on air at all --
                # the standing service numbers included, not just the
                # editorial slot the topic editor already respects. The weather
                # resolved facts (tense states, daylight phase) are part of
                # this branch of content and honor the same switch.
                for key in ("current_weather", "high_state",
                            "overnight_low_state", "daylight_phase",
                            "daylight_line"):
                    service.pop(key, None)
                extras.pop("weather_context", None)
            if service.get("current_weather"):
                # The numbers ride the standing service channel; the
                # editorial weather slot would only duplicate them.
                extras.pop("weather_context", None)
            # Service presence: time and weather do not share a
            # cadence. The clock is due unless one ACTUALLY AIRED within
            # the gap (a news intro's clock counts); weather is due 25
            # minutes after it actually aired or at once on a material
            # change; daylight is withheld while unchanged inside the
            # cadence window and flagged when its phase has moved. A
            # concept that is not due is removed -- never offered, so it
            # can neither air nor be recorded. The ledger is break
            # telemetry (clock_aired / service_aired); nothing new persists.
            #
            # The reference instant is air_at -- WHEN THESE WORDS WILL BE
            # HEARD, the same projection the clock itself is written to --
            # not now_dt, which is up to RENDER_LEAD_S earlier and would
            # understate every gap by that lead (a clock would stay
            # withheld for minutes after its gap had really elapsed).
            # Without a trustworthy projection the render instant is the
            # honest fallback; service_facts omits the clock there anyway.
            try:
                import service_presence
                state = service_presence.ledger(queue_id=queue_id,
                                                db_path=db_path)
                service_aired = service_presence.apply(
                    service, state, air_at or now_dt,
                    _service_cooldown_seconds())
                # exactly the facts the writer is about to be given; what it
                # does with them is decided after the break is written
                service_offered = dict(service)
            except Exception:  # noqa: BLE001
                logger.warning("service presence policy failed (facts kept)",
                               exc_info=True)
                service.pop("service_sig", None)
                service_aired = {}
            context.update(extras)
            context.update(service)
            if air_at is not None:
                context["estimated_air_at"] = _iso(air_at)
            # Song context: grounded album facts + derived character
            # for both sides of the break, and -- only when the sparse
            # deterministic gate opens -- a short semantic note about the
            # ending track. Fenced separately: song knowledge must never
            # cost a break.
            try:
                import song_context
                _enrich_song_context(context, session, service, chattiness,
                                     now_dt, db_path)
            except Exception:
                logger.warning("song context enrichment failed "
                               "(ordinary break)", exc_info=True)
        except Exception:
            logger.warning("dj context enrichment failed (music-only break)",
                           exc_info=True)
        # False on every production render since the decoupling: only the
        # internal LOW policy sets it. See _quiet_slot.
        if chat_policy.get("quiet_slots") and _quiet_slot(
                token, queue_id, brk, session, extras, now_dt, db_path,
                calendar_event=context.get("calendar_event")):
            return "quiet_slot"
        import playback
        resolved = playback.resolved_session_tracks(session["session_id"],
                                                    db_path=db_path)
        if scheduled:
            # Calendar announcements: a due plain ANNOUNCE rides THIS
            # break and is handed to the writer as required content.
            _bind_announcement_carrier(token, queue_id, context, now_dt,
                                       deps, db_path)
        story = (_music_story_stage(token, queue_id, planner_boundary,
                                    context, resolved, now_dt, deps, db_path)
                 if scheduled else None)
        if story is not None:
            # A prepared music story hosts this break: no planner call, no
            # model call at air time (split_hosting / dj_memory).
            context["story_break"] = story
            _announcement_room(context)
        elif planner_active:
            # Smart DJ (smart_dj): one bounded, grounded planner decision
            # per eligible boundary -- speak, or say nothing. Manual
            # intents skip it: the listener asked for this break.
            if getattr(planner_fn, "writer_style", None) == smart_dj.WRITER_P2:
                # P2's qualified room: the prompt's ceiling, the cleaner's
                # cap, required-hosting completion and the floor all read
                # this one value for THIS break (smart_dj.P2_MAX_WORDS).
                context["max_words"] = smart_dj.P2_MAX_WORDS
                context["writer_style"] = smart_dj.WRITER_P2
            _announcement_room(context)
            outcome = _smart_dj_stage(token, queue_id, planner_boundary,
                                      context, resolved, service_offered,
                                      planner_fn, now_dt, db_path,
                                      next_decision_at=planner_next_at)
            if outcome is not None:
                return outcome
        else:
            _announcement_room(context)
        ok = render_break(token, session, resolved, None,
                          llm_fn=deps.get("llm_fn"),
                          tts_fn=deps.get("tts_fn"),
                          transcode_fn=deps.get("transcode_fn"),
                          context=context, now=now_dt, db_path=db_path,
                          render_deadline=render_deadline)
        if not ok:
            return "render_failed"
        if context.get("semantic_themes"):
            # The cooldown/repetition ledger IS break telemetry -- no
            # bespoke state file.
            _note(token, {"semantic_themes": context["semantic_themes"],
                          "semantic_tone": context.get("semantic_tone"),
                          "semantic_reason": context.get("semantic_reason"),
                          "semantic_used_at": _iso(now_dt)}, db_path)
        if service_aired:
            # The cooldown ledger records what the
            # LISTENER HEARD, not what the writer was offered. Weather and
            # time are optional background the writer often (correctly)
            # omits; counting an unspoken fact as aired suppressed it for a
            # further window it never used. Read the final accepted text --
            # after the writer's repair and the deterministic callback
            # composition -- and keep only the concepts genuinely spoken.
            try:
                aired_brk = get_break(token, db_path) or {}
                final_text = aired_brk.get("spoken_text") or ""
                heard = dj_context.spoken_service_concepts(
                    final_text, service_offered)
                # atmosphere records what it SAID: a code-owned daylight line
                # the render confirmed in the text states the offered phase,
                # whether or not the detector's vocabulary knows its words
                atm = (aired_brk.get("telemetry") or {}).get("atmosphere") or {}
                if atm.get("aired") and atm.get("records") == "daylight" and \
                        atm.get("text") and atm["text"] in final_text:
                    heard = set(heard) | {"daylight"}
                service_aired = {k: v for k, v in service_aired.items()
                                 if k in heard}
                # the supplied clock the text stated (exact or an equivalent
                # form): the fact the five-minute time gap is measured from
                clock = dj_context.spoken_clock(final_text, service_offered)
                if clock:
                    _note(token, {"clock_aired": clock}, db_path)
            except Exception:  # noqa: BLE001
                logger.warning("spoken service check failed (recording "
                               "nothing)", exc_info=True)
                service_aired = {}
        if service_aired:
            _note(token, {"service_aired": service_aired}, db_path)
        if calendar_marks and context.get("calendar_event"):
            # One band, one mention -- but only a mention that happened.
            # The same anchor test a cited fact gets: the event's title
            # has to be in the words the listener will hear. A break that
            # was offered the calendar and wrote about the music instead
            # leaves the band untouched for the next one.
            try:
                aired = (get_break(token, db_path) or {}).get(
                    "spoken_text") or ""
                if smart_dj.calendar_spoken(
                        aired, {"calendar": context["calendar_event"]}):
                    marks = {**(marks or {}), **calendar_marks}
            except Exception:  # noqa: BLE001
                logger.warning("calendar mark check failed (band kept)",
                               exc_info=True)
        if context.get("knowledge_facts") and context.get("smart_dj"):
            # FACT NOVELTY COMMITS HERE AND NOWHERE ELSE. Not when the
            # pack was queried, not when the fact was offered, not when
            # the planner chose FACT, not when text was generated and
            # then rejected -- only when the copy that will actually play
            # cites the fact AND says what it is about. Anything less
            # would retire a fact for a week that no listener ever heard.
            try:
                aired = (get_break(token, db_path) or {}).get(
                    "spoken_text") or ""
                import music_knowledge as _mk
                fired = _mk.aired_fact_ids(
                    context["knowledge_facts"],
                    list((context["smart_dj"].get("fact_ids")) or []), aired)
                if fired:
                    marks = {**(marks or {}), "facts": fired}
                    _note(token, {"knowledge_aired": fired}, db_path)
            except Exception:  # noqa: BLE001
                logger.warning("knowledge fact mark check failed (fact "
                               "stays eligible)", exc_info=True)
        if marks:
            # Suppression bookkeeping commits ONLY for the renderer that won
            # the atomic claim and produced a READY break.
            try:
                import dj_context
                dj_context.commit_marks(marks, now_dt)
            except Exception:
                logger.warning("dj context mark commit failed", exc_info=True)
        brk = get_break(token, db_path)
        try:
            state = read_state(queue_id)  # re-read: rendering took real time
        except Exception:
            return "no_state"
    if brk is not None and brk["state"] == "ready":
        _attach_successor_window(brk, state, queue_id, deps.get("items_fn"))
        if not context_still_valid(brk, state):
            mismatch = context_mismatch(brk, state) or (INSERT_OTHER, None)
            _trace_insert(token, mismatch[0], state, now_dt, db_path,
                          detail=mismatch[1], stage="pre_insert")
            if break_purpose(brk.get("telemetry")) == PURPOSE_MANUAL:
                # The state was re-read after rendering: the pin may have
                # broken meanwhile (retarget/terminate), or only the
                # successor changed (pin holds: discard the stale render and
                # re-render against the live queue -- no retarget consumed,
                # the pinned boundary is still the target).
                outcome = _manual_pin_check(brk, state, now_dt, db_path)
                if outcome:
                    return outcome
                tel = brk.get("telemetry") or {}
                keep = {k: tel[k] for k in
                        (*_MANUAL_KEEP_KEYS, "intended_boundary_queue_item_id",
                         "intended_boundary_name", "target_air_at",
                         "retarget_count", "retargeted_at", "retargeted_from")
                        if k in tel}
                if _reset_manual_row(token, keep, now_dt, db_path):
                    return "manual_rerender"
                return "idle"
            revoke(token, now_dt, db_path, reason="stale_context")
            plan_break(queue_id, session, now_dt, db_path)
            return "stale_revoked"
        # Speech may have been switched off while THIS pass was rendering
        # (the check above ran before the render began). Ask again at the
        # last moment an ordinary break can still be kept out of the queue.
        if (is_ordinary_break(brk.get("telemetry"))
                and not dj_speech_enabled()):
            if _close_for_speech(token, queue_id, brk, session, now_dt,
                                 db_path):
                return SPEECH_DISABLED
            return "idle"
        carried = (brk.get("telemetry") or {}).get("announcement")
        before_claim = None
        if carried:
            import announcements
            before_claim = announcements.carrier_before_claim(token, now_dt)
            if before_claim is None:
                # The occurrence no longer names this break (or its store
                # cannot be read): its words must not air unrecorded.
                # Re-render without them at the next boundary.
                revoke(token, now_dt, db_path,
                       reason="announcement_carrier_unbound")
                plan_break(queue_id, session, now_dt, db_path)
                return "stale_revoked"
        result = try_insert(token, now_dt, state_fn=deps.get("state_fn"),
                            insert_fn=deps.get("insert_fn"), db_path=db_path,
                            items_fn=deps.get("items_fn"),
                            before_claim=before_claim)
        if carried:
            announcements.carrier_after_insert(token, result, db_path)
            if result == "handoff_unrecorded":
                revoke(token, now_dt, db_path,
                       reason="announcement_handoff_unrecorded")
                plan_break(queue_id, session, now_dt, db_path)
                return "stale_revoked"
        if result == "stale":
            # The same answer a broken adjacency gets above, reached one
            # step later: discard the prepared break and plan again from
            # live context. plan_break only creates an INTENT, so if there
            # is no runway left this simply produces nothing and the music
            # plays on -- a break is skipped, never forced.
            revoke(token, now_dt, db_path, reason="stale_pre_air")
            replacement = plan_break(queue_id, session, now_dt, db_path)
            if (replacement and stale_clock is not None and adopted_ready
                    and ((get_break(token, db_path) or {}).get("telemetry")
                         or {}).get("pre_air_refusal") == "stale_clock"):
                # Only the clock was wrong: this boundary is still the one
                # the break was written for, and nothing has used it. The
                # caller decides whether the replacement can take it now.
                # Never after a render in this same pass: one render per
                # pass, and that render's clock was set by this pass's now.
                stale_clock.update(stale=token, replacement=replacement)
            return "stale_revoked"
        return result
    return "idle"


# --- insertion telemetry (OBSERVABILITY ONLY) ---------------------------------
#
# A break can be rendered, sit READY for a minute or two before its
# boundary -- as long as breaks that do air -- and never be inserted, then
# be revoked as stale_context one boundary later. Without these records
# nothing says WHY: a closed gate returns 'gate_closed' silently on
# every tick, so the one question that matters (when did MA close the Flow
# gate relative to the boundary, and was the break even attempted?) has no
# evidence. These records are that evidence. They decide nothing: every
# call site already had its outcome before the record is written, the
# writers are fenced, and no return value, retry or queue action depends
# on them.
#
# One record per DISTINCT observation, not per tick: a gate that stays
# closed across twenty polls is one `insert_gate_closed` line whose
# successor carries `repeats_before: 19`, and the repeats cost no writes.
# The break row keeps a running summary (`insert_trace`), updated with each
# record, so the counts survive even when forensics is off. No queue
# contents, names, URLs or credentials are recorded -- queue item ids,
# indexes and seconds only.

INSERT_SUCCESS = "insert_success"
INSERT_GATE_CLOSED = "insert_gate_closed"
INSERT_REJECTED = "insert_rejected"            # pre-air refusal (e.g. clock)
INSERT_SUCCESSOR_MISMATCH = "successor_mismatch"
INSERT_HEAD_CHANGED = "head_changed"
INSERT_QUEUE_UNREADABLE = "queue_window_unreadable"
INSERT_MA_FAILURE = "ma_api_failure"
INSERT_LIFECYCLE_UNSAFE = "lifecycle_unsafe"
INSERT_OTHER = "other"                         # another fail-closed reason
#: Outcomes that end this break's insertion story: always recorded.
_INSERT_TERMINAL = frozenset({INSERT_SUCCESS, INSERT_REJECTED,
                              INSERT_SUCCESSOR_MISMATCH, INSERT_HEAD_CHANGED,
                              INSERT_MA_FAILURE, INSERT_LIFECYCLE_UNSAFE})


def _insert_snapshot(brk: dict[str, Any], state: dict[str, Any] | None
                     ) -> dict[str, Any]:
    """The compact gate/adjacency picture at one insertion decision."""
    tel = brk.get("telemetry") or {}
    st = state if isinstance(state, dict) else {}
    cur_index = st.get("current_index")
    items = st.get("next_items")
    if not isinstance(items, list):
        window = "absent"                  # no window attached: next_item only
    else:
        window = len(items)                # 0 = attached, nothing placeable
    try:
        nxt = next_music_item(st).get("queue_item_id") if st else None
    except Exception:  # noqa: BLE001
        nxt = None
    ttb = None
    try:
        rem = _remaining_s(st) if st else None
        ttb = round(rem, 1) if rem is not None else None
    except Exception:  # noqa: BLE001
        ttb = None
    return {
        "want_pre": tel.get("intended_pre_queue_item_id"),
        "want_pre_index": tel.get("intended_pre_index"),
        "want_post": tel.get("intended_post_queue_item_id"),
        "head": (st.get("current_item") or {}).get("queue_item_id"),
        "cur_index": cur_index,
        "buf_index": st.get("index_in_buffer"),
        "target_pos": (cur_index + 1 if isinstance(cur_index, int)
                       else None),
        "next_music": nxt,
        "window": window,
        "player": st.get("state"),
        "ttb_s": ttb,
    }


#: Identical observations since the last record, per break -- held in
#: memory so a gate that stays closed for twenty ticks costs no writes
#: (test_storage_writes); folded into the row at the next record. Bounded:
#: a break that never produces another record is simply forgotten.
_insert_seen: dict[str, dict[str, Any]] = {}
_INSERT_SEEN_MAX = 64


def _trace_insert(token: str, outcome: str, state: dict[str, Any] | None,
                  now: datetime, db_path=None, *, detail: str | None = None,
                  stage: str = "try_insert") -> None:
    """Record one insertion decision for a prepared break. Never raises and
    never changes anything but telemetry."""
    try:
        brk = get_break(token, db_path)
        if brk is None:
            return
        snap = _insert_snapshot(brk, state)
        key = [outcome, detail, stage, snap["head"], snap["cur_index"],
               snap["buf_index"], snap["next_music"], snap["window"]]
        seen = _insert_seen.get(token)
        if (outcome not in _INSERT_TERMINAL and seen is not None
                and seen["key"] == key):
            seen["repeats"] += 1
            return
        repeats = seen["repeats"] if seen else 0
        tel = brk.get("telemetry") or {}
        trace = dict(tel.get("insert_trace") or {})
        by = dict(trace.get("by_outcome") or {})
        if repeats and seen:
            by[seen["key"][0]] = int(by.get(seen["key"][0]) or 0) + repeats
        by[outcome] = int(by.get(outcome) or 0) + 1
        attempts = int(trace.get("attempts") or 0) + repeats + 1
        rec = {"outcome": outcome, "stage": stage, **snap}
        if detail:
            rec["detail"] = detail
        trace.update({"attempts": attempts, "by_outcome": by,
                      "first_at": trace.get("first_at") or _iso(now),
                      "last_at": _iso(now), "last_outcome": outcome,
                      "last": rec})
        _note(token, {"insert_trace": trace}, db_path)
        if outcome in _INSERT_TERMINAL:
            _insert_seen.pop(token, None)
        else:
            if len(_insert_seen) >= _INSERT_SEEN_MAX:
                _insert_seen.pop(next(iter(_insert_seen)))
            _insert_seen[token] = {"key": key, "repeats": 0}
        logger.info("dj break %s insert %s%s (stage=%s head=%s cur=%s "
                    "buf=%s next_music=%s window=%s ttb=%s)",
                    token[:8], outcome, f"/{detail}" if detail else "",
                    stage, snap["head"], snap["cur_index"], snap["buf_index"],
                    snap["next_music"], snap["window"], snap["ttb_s"])
        import dj_forensics as _fx
        _fx.append({"at": _iso(now), "event": "insert", "break_id": token,
                    "queue_id": brk.get("queue_id"), "attempt": attempts,
                    "repeats_before": repeats,
                    "target_air_at": tel.get("target_air_at"), **rec})
    except Exception:  # noqa: BLE001 - diagnostics never cost a break
        logger.debug("insert trace failed for %s", token[:8], exc_info=True)


def _trace_passed_over(queue_id: str, outcome: str, detail: str,
                       state: dict[str, Any] | None, now: datetime,
                       db_path=None, exclude: str | None = None) -> None:
    """A READY break that this pass never offered to try_insert -- the pass
    returned first (news block, unreadable queue) or chose a newer intent.
    Invisible until now: those breaks left no attempt at all. Read-only
    lookup; never raises."""
    try:
        c = _connect(db_path)
        try:
            rows = c.execute(
                "SELECT token FROM dj_breaks WHERE queue_id = ? AND "
                "state = 'ready' AND token != ?",
                (queue_id, exclude or "")).fetchall()
        finally:
            c.close()
        for (tok,) in rows:
            _trace_insert(tok, outcome, state, now, db_path, detail=detail,
                          stage="pass")
    except Exception:  # noqa: BLE001
        logger.debug("passed-over trace failed", exc_info=True)


# --- committed handoffs --------------------------------------------------------
#
# A break handed to Music Assistant is finished audio. When it says what
# comes next ("up next, X"), X is a PROMISE the queue has to keep until the
# break has aired: nothing Parlorwave does may delete X first. A Mode 2
# station switch could: the switch retires the old
# station's songs by id, and X is usually one of them.
#
# THE EVIDENCE IS STRUCTURED, NEVER THE PROSE. _context_from_state records
# the next MUSIC item's exact Music Assistant queue_item_id; render_break
# keeps it as `intended_post_queue_item_id` only when the writer was allowed
# to introduce that song; context_mismatch proves it is still the next
# music immediately before the ready->queued claim. From the claim until
# the break completes (the next music starting ends it), that queue item is
# committed. A duplicate recording, the same title or the same URI queued
# again is a different queue item and is not.

#: Break states whose audio Music Assistant holds and has not finished.
COMMITTED_STATES = ("queued", "airing")

#: Verifying a successor and claiming the row (try_insert) is one step
#: against the station switch's check-then-delete (station_live). Held only
#: around those short sections, never across a render or an MA append.
HANDOFF_LOCK = threading.Lock()
HANDOFF_WAIT_S = 3.0


def _successor_claim(brk: dict[str, Any] | None) -> tuple[bool, str | None]:
    """(does this break name a next song, the exact queue item it names).

    A name recorded without a usable id is a promise whose object cannot be
    proven: (True, None)."""
    tel = (brk or {}).get("telemetry")
    if not isinstance(tel, dict):
        return False, None
    qiid = tel.get("intended_post_queue_item_id")
    qiid = qiid if isinstance(qiid, str) and qiid else None
    return bool(qiid or tel.get("intended_post_name")), qiid


def committed_handoffs(queue_id: str, db_path=None) -> dict[str, Any]:
    """The queue items breaks already given to Music Assistant have
    promised to hand off to, from the durable rows alone (so a restart
    loses nothing):

      ids        exact queue_item_ids -- keep every one of them
      unproven   short tokens of committed breaks that name a next song
                 without a usable id, or whose record cannot be read: the
                 caller cannot know what they promised and must delete
                 nothing

    A committed break that names no next song promises nothing. Raises
    sqlite3.Error when the rows cannot be read; the caller treats that as
    unproven too."""
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT token, telemetry FROM dj_breaks WHERE queue_id = ? "
            "AND state IN (%s)" % ",".join("?" * len(COMMITTED_STATES)),
            (queue_id, *COMMITTED_STATES)).fetchall()
    finally:
        c.close()
    ids: set[str] = set()
    unproven: list[str] = []
    for token, raw in rows:
        try:
            tel = json.loads(raw or "{}")
        except (TypeError, ValueError):
            tel = None
        if not isinstance(tel, dict):
            unproven.append(str(token)[:8])
            continue
        named, qiid = _successor_claim({"telemetry": tel})
        if qiid:
            ids.add(qiid)
        elif named:
            unproven.append(str(token)[:8])
    return {"ids": ids, "unproven": unproven}


def _verify_and_claim(brk: dict[str, Any], token: str, now_dt: datetime,
                      db_path, read_state, items_fn, before_claim
                      ) -> tuple[str, dict[str, Any] | None]:
    """try_insert's gate, pre-air verification and ready->queued claim:
    ('claimed', state) or (try_insert's answer, state)."""
    import ma_queue
    try:
        _stamp(token, "queue_read", now_dt, db_path)
        state = read_state(brk["queue_id"])
        if not ma_queue.insertion_gate_open(state):
            unknown = (not isinstance(state, dict)
                       or state.get("current_index") is None
                       or state.get("index_in_buffer") is None)
            _trace_insert(token, INSERT_QUEUE_UNREADABLE if unknown
                          else INSERT_GATE_CLOSED, state, now_dt, db_path,
                          detail="gate_index_unknown" if unknown else None)
            return "gate_closed", state
    except Exception as exc:  # unreadable queue = closed gate, never a guess
        logger.warning("dj break %s: queue state unreadable: %s", token, exc)
        _trace_insert(token, INSERT_QUEUE_UNREADABLE, None, now_dt, db_path,
                      detail=f"queue_read:{type(exc).__name__}")
        return "gate_closed", None
    _attach_successor_window(brk, state, brk["queue_id"], items_fn)
    ok, why = pre_air_revalidate(brk, state, _projected_air_local(state, now_dt))
    if not ok:
        logger.info("dj break %s not aired (%s): its assumptions stopped "
                    "being true while it waited", token[:8], why)
        _note(token, {"pre_air_refusal": why}, db_path)
        mismatch = (context_mismatch(brk, state) if why == "stale_context"
                    else None)
        if mismatch:
            _trace_insert(token, mismatch[0], state, now_dt, db_path,
                          detail=mismatch[1])
        else:
            _trace_insert(token, INSERT_REJECTED, state, now_dt, db_path,
                          detail=why)
        return "stale", state
    if before_claim is not None:
        try:
            recorded = bool(before_claim())
        except Exception:  # noqa: BLE001 - no evidence, no claim
            logger.warning("dj break %s: pre-claim evidence not recorded; "
                           "not inserted", token[:8], exc_info=True)
            recorded = False
        if not recorded:
            _trace_insert(token, INSERT_OTHER, state, now_dt, db_path,
                          detail="handoff_unrecorded")
            return "handoff_unrecorded", state
    if not _transition(token, ("ready",), "queued", now_dt, db_path):
        # someone else won the claim -- exactly once wins
        _trace_insert(token, INSERT_LIFECYCLE_UNSAFE, state, now_dt, db_path,
                      detail="claim_lost")
        return "not_ready", state
    return "claimed", state


def try_insert(token: str, now: datetime | None = None,
               state_fn: Callable[[str], dict[str, Any]] | None = None,
               insert_fn: Callable[..., None] | None = None,
               db_path=None,
               items_fn: Callable | None = None,
               before_claim: Callable[[], bool] | None = None) -> str:
    """Attempt to queue a READY break at the current boundary.

    Returns one of: 'queued', 'gate_closed', 'not_ready', 'stale', 'failed'
    -- and, only for a caller that passes `before_claim`,
    'handoff_unrecorded'.

    `before_claim` runs after every gate has passed and IMMEDIATELY before
    the ready->queued claim. It lets a caller write durable evidence that
    this row may be about to reach Music Assistant; if it returns False or
    raises, nothing is claimed and nothing is inserted (the row stays
    ready). Announcements use it so that "no evidence was written" can
    only ever mean "never handed to MA" -- the basis of their no-replay
    contract. Every other caller omits it and is unaffected.

    A failed enqueue is noted in telemetry as `enqueue_failure`:
    'not_delivered' when ma_queue.enqueue_never_delivered can prove MA
    never received the request, else 'uncertain'.

    The structural Flow-mode gate (index_in_buffer == current_index) is the
    ONLY insertion condition. A closed gate means the boundary is missed:
    the break stays 'ready' and simply retargets the next eligible boundary
    (until the TTL revokes it). No transport action is ever taken on a
    missed boundary. The ready->queued atomic claim happens BEFORE the
    enqueue call, so a concurrent duplicate can never double-insert; an
    enqueue failure then lands in 'failed' (music untouched).

    'stale' means the break's own assumptions stopped being true while it
    waited (see pre_air_revalidate). It is checked HERE, against the same
    queue read the gate uses and immediately before the claim, because this
    is the last instant at which anything is still recallable -- after the
    claim the audio belongs to Music Assistant. The row is left 'ready' for
    the caller to revoke and replan; nothing is inserted, so music is
    untouched either way."""
    import ma_queue
    now_dt = now or service_clock.legacy_now()
    brk = get_break(token, db_path)
    if brk is None or brk["state"] != "ready":
        if brk is not None:
            _trace_insert(token, INSERT_LIFECYCLE_UNSAFE, None, now_dt,
                          db_path, detail=f"state:{brk['state']}")
        return "not_ready"
    read_state = state_fn or ma_queue.queue_state
    do_insert = insert_fn or ma_queue.insert_next
    # A break that names the song coming next is about to make a PROMISE
    # (see committed_handoffs). Verifying that song and claiming the row is
    # one step against a station switch deleting it: the switch holds the
    # same lock around each delete. A break that names nothing promises
    # nothing and never waits.
    promises = _successor_claim(brk)[0]
    if promises and not HANDOFF_LOCK.acquire(timeout=HANDOFF_WAIT_S):
        _trace_insert(token, INSERT_OTHER, None, now_dt, db_path,
                      detail="handoff_busy")
        return "gate_closed"
    try:
        outcome, state = _verify_and_claim(brk, token, now_dt, db_path,
                                           read_state, items_fn,
                                           before_claim)
    finally:
        if promises:
            HANDOFF_LOCK.release()
    if outcome != "claimed":
        return outcome
    try:
        _stamp(token, "enqueue", service_clock.legacy_now(), db_path)
        do_insert(brk["queue_id"], audio_url(token), BREAK_ITEM_TITLE, BREAK_ITEM_ARTIST)
    except Exception as exc:  # noqa: BLE001
        logger.warning("dj break %s enqueue failed: %s", token, exc)
        try:
            delivered = ("not_delivered"
                         if ma_queue.enqueue_never_delivered(exc)
                         else "uncertain")
        except Exception:  # noqa: BLE001 - unknown is uncertain
            delivered = "uncertain"
        _note(token, {"enqueue_failure": delivered}, db_path)
        _transition(token, ("queued",), "failed", service_clock.legacy_now(), db_path,
                    error=str(exc)[:300])
        _trace_insert(token, INSERT_MA_FAILURE, state, now_dt, db_path,
                      detail=type(exc).__name__)
        return "failed"
    _trace_insert(token, INSERT_SUCCESS, state, now_dt, db_path)
    return "queued"


# --- station-request acknowledgement (event liner) ----------------------------

def create_request_ack(queue_id: str, request: dict[str, Any],
                       session_id: str = "", now: datetime | None = None,
                       db_path=None) -> str:
    """An acknowledgement INTENT for a station request, as an ordinary
    dj_breaks row: same token, audio serving, lifecycle, claims, retention
    and airing evidence as any break. Only the telemetry differs -- purpose
    marks it a liner (excluded from the cadence anchor) and request carries
    the grounded facts the liner may use, including the queue_item_id the
    request was made against, which the freshness gate re-checks before any
    enqueue. No target_air_at: a liner answers THIS boundary or none."""
    now_dt = now or service_clock.legacy_now()
    token = _secrets.token_hex(16)
    tel = {"intent_created": _iso(now_dt),
           "purpose": PURPOSE_STATION_REQUEST,
           "request": {k: request.get(k) for k in
                       ("artist", "song", "first_new_track", "queue_item_id")}}
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO dj_breaks (token, queue_id, session_id, state, "
            "created_at, updated_at, telemetry) "
            "VALUES (?, ?, ?, 'planned', ?, ?, ?)",
            (token, queue_id, session_id or "", _iso(now_dt), _iso(now_dt),
             json.dumps(tel)))
        c.commit()
    finally:
        c.close()
    return token


NEWS_INTRO_MAX_WORDS = 25
# A breath between the intro and the bulletin. Ours to shape: this pads OUR
# rendered intro, never the publisher's bytes.
NEWS_INTRO_TAIL_S = float(os.environ.get("NEWS_INTRO_TAIL_S", "0.6"))


def create_news_intro(queue_id: str, news: dict[str, Any],
                      session_id: str = "", now: datetime | None = None,
                      db_path=None) -> str:
    """An intent for the line that introduces a publisher bulletin.

    It is an ordinary dj_breaks row -- same token, audio serving, claims,
    lifecycle, retention and airing evidence as any break -- because it IS
    ParlorWave speech. Only the telemetry differs: purpose marks it a liner
    (cadence-exempt), and the news_* fields link it to the publisher event
    so the two lifecycles can be reconciled from either side.

    The BULLETIN never becomes a dj_breaks row. The intro is ParlorWave
    programming; the bulletin is publisher programming with its own
    artifact and its own one-slot cache."""
    now_dt = now or service_clock.legacy_now()
    token = _secrets.token_hex(16)
    tel = {"intent_created": _iso(now_dt),
           "purpose": PURPOSE_NEWS_INTRO,
           "news_token": news.get("news_token"),
           "news_target_hour": news.get("target_hour"),
           "news_target_at": news.get("target_at"),
           "estimated_air_at": news.get("estimated_air_at")}
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO dj_breaks (token, queue_id, session_id, state, "
            "created_at, updated_at, telemetry) "
            "VALUES (?, ?, ?, 'planned', ?, ?, ?)",
            (token, queue_id, session_id or "", _iso(now_dt), _iso(now_dt),
             json.dumps(tel)))
        c.commit()
    finally:
        c.close()
    return token


def render_news_intro(token: str, text: str, deps: dict[str, Any] | None = None,
                      now: datetime | None = None, db_path=None) -> bool:
    """Render a DETERMINISTIC intro line through the ordinary finite-audio
    path -- same TTS provider, same speech.normalize_for_tts copy, same
    MP3 contract, same ID3, same ready/failed states.

    The text is supplied, not generated: render_break's llm_fn seam is fed
    a function that returns it verbatim, so no prompt is built and no model
    is called. That is the whole point of a template line."""
    deps = deps or DEFAULT_DEPS or {}
    ok = render_break(
        token, {"session_id": (get_break(token, db_path) or {}).get(
            "session_id", "")}, [], None,
        llm_fn=lambda _prompt: text,
        tts_fn=deps.get("tts_fn"), transcode_fn=deps.get("transcode_fn"),
        pad_fn=deps.get("pad_fn"),
        context={"purpose": PURPOSE_NEWS_INTRO,
                 "max_words": NEWS_INTRO_MAX_WORDS,
                 "trailing_silence_s": NEWS_INTRO_TAIL_S},
        now=now, db_path=db_path)
    if ok:
        # A template clock is a real time check the listener heard: record
        # it so the next music break's clock respects the five-minute gap.
        import service_presence
        m = service_presence._CLOCK_RE.search(text or "")
        if m:
            _note(token, {"clock_aired": m.group(0)}, db_path)
    return ok


# A generous ceiling: announcement copy is bounded by the parser's payload
# cap, and the render must never truncate operator-approved wording.
ANNOUNCEMENT_MAX_WORDS = 150
# The same breath a news intro gets between speech and music.
ANNOUNCEMENT_TAIL_S = float(os.environ.get("ANNOUNCEMENT_TAIL_S", "0.6"))


def create_announcement_break(queue_id: str, occurrence: dict[str, Any],
                              session_id: str = "", origin: str = "",
                              now: datetime | None = None,
                              db_path=None,
                              voice: dict[str, Any] | None = None) -> str:
    """An intent for one calendar announcement occurrence, as an ordinary
    dj_breaks row -- same token, audio serving, claims, lifecycle,
    retention, tombstones and airing evidence as any break. Only the
    telemetry differs: purpose marks it an event liner (cadence-exempt),
    and the announcement block links it to the occurrence so the two
    lifecycles reconcile from either side (announcements.py adopts this
    row's outcome; Air Check names the occurrence).

    `voice` is the occurrence's already-decided
    voice request: what the calendar asked for, the AUTHORIZED voice id it
    resolved to (or why it did not), and the station's default announcement
    voice at commit time. It is written into the row so the render reads a
    durable decision, never memory, and so the evidence survives a restart
    between commit and render."""
    now_dt = now or service_clock.legacy_now()
    token = _secrets.token_hex(16)
    tel = {"intent_created": _iso(now_dt),
           "purpose": PURPOSE_ANNOUNCEMENT,
           "announcement": {
               "occurrence_id": occurrence.get("id"),
               "source": occurrence.get("source"),
               "start": occurrence.get("start"),
               "mode": occurrence.get("mode"),
               "origin": origin,
               "payload_sha": occurrence.get("payload_sha"),
           }}
    if voice:
        tel["announcement"].update({
            "voice_requested": voice.get("requested"),
            "voice_id": voice.get("voice_id"),
            "voice_identity": voice.get("voice_identity"),
            "voice_reason": voice.get("reason"),
            "default_voice_id": voice.get("default_voice_id"),
            "default_voice_identity": voice.get("default_voice_identity"),
        })
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO dj_breaks (token, queue_id, session_id, state, "
            "created_at, updated_at, telemetry) "
            "VALUES (?, ?, ?, 'planned', ?, ?, ?)",
            (token, queue_id, session_id or "", _iso(now_dt), _iso(now_dt),
             json.dumps(tel)))
        c.commit()
    finally:
        c.close()
    return token


def render_announcement(token: str, text: str,
                        deps: dict[str, Any] | None = None,
                        now: datetime | None = None, db_path=None) -> bool:
    """Render FINISHED announcement copy through the ordinary finite-audio
    path -- same TTS provider, same speech.normalize_for_tts pass, same
    MP3 contract, same ready/failed states.

    The words were decided upstream (operator-approved EXACT copy, an
    already-validated Smart rewrite, or the deterministic renderer) and
    render_break's announcement branch carries them VERBATIM: no writer
    runs, no guard rewrites, no clock/daylight normalization touches
    them, and nothing records a clock claim -- an event's "5 PM" is never
    the station telling the time."""
    deps = deps or DEFAULT_DEPS or {}
    ann = (get_break(token, db_path) or {}).get(
        "telemetry", {}).get("announcement") or {}
    return render_break(
        token, {"session_id": (get_break(token, db_path) or {}).get(
            "session_id", "")}, [], None,
        llm_fn=lambda _prompt: text,
        # WHO speaks is the only thing the voice request may change. The
        # text reaching this seam is the same normalize_for_tts copy every
        # finite render sends; the seam never sees, and cannot alter, the
        # words it is handed.
        tts_fn=_announcement_tts(token, ann, deps, db_path),
        transcode_fn=deps.get("transcode_fn"),
        pad_fn=deps.get("pad_fn"),
        context={"purpose": PURPOSE_ANNOUNCEMENT,
                 "max_words": ANNOUNCEMENT_MAX_WORDS,
                 "trailing_silence_s": ANNOUNCEMENT_TAIL_S,
                 "announcement": ann},
        now=now, db_path=db_path)


# Announcement voices: why an announcement was not spoken by the voice
# its event asked for. `announcement_voice_fallback` is the render-time
# case -- an AUTHORIZED voice that could not speak just now; the two
# voice_library.REJECT_* codes are commit-time refusals of the request
# itself. All three land in the row, the occurrence and dj_forensics.
ANNOUNCEMENT_VOICE_FALLBACK = "announcement_voice_fallback"
#: Where the voice that finally spoke came from.
VOICE_SOURCE_EVENT = "event"        # the calendar's authorized request
VOICE_SOURCE_DEFAULT = "default"    # station.announcements.default_voice_id
VOICE_SOURCE_STATION = "station"    # the DJ/persona voice: the floor


def _announcement_tts(token: str, ann: dict[str, Any],
                      deps: dict[str, Any], db_path=None
                      ) -> Callable[[str], bytes]:
    """The speech seam for ONE announcement render: a `tts_fn` that tries
    the voices this announcement may be spoken with, most specific first,
    and records which one spoke and why the others did not.

    Precedence:
      1. the event's request, when it resolved to an AUTHORIZED voice
         (announcement.voice_id -- decided at commit, durable in the row);
      2. the station's default announcement voice
         (announcement.default_voice_id, from settings at commit);
      3. the station's own DJ/persona voice -- deps' tts_fn or default_tts,
         exactly the call every announcement made before this seam. The
         DJ voice is the legacy delivery floor under its ordinary DJ
         semantics, not a selected announcement voice, and is never asked
         for an announcement scope.

    RENDER-TIME REVALIDATION. A commit's decision does not outlive the
    registry it was made against. Before either named voice is spoken it
    is checked again, against the registry AS IT IS NOW
    (voice_library.check_announcement_voice): the id still names a
    voice; a custom slot still holds the SAME recording the commit saw
    (its fingerprint), so a slot re-enrolled for somebody else never reads
    an older occurrence in the new person's voice; and the CURRENT scope
    still permits announcements -- for the event's voice AND the default,
    because dj.voice_scopes is what a voice may do and default_voice_id
    only which authorized voice is preferred: naming a default grants
    nothing. The default is additionally required to still BE the
    configured default. A failed check is an attempt with its reason (voice_missing,
    voice_authorization_withdrawn, voice_identity_changed) and nothing is
    synthesized with that voice.

    A voice that cannot speak (engine down, sample deleted, provider off,
    empty audio) costs nothing but a recorded reason: the SAME text goes
    to the next voice. A safety reminder is never dropped for lack of its
    preferred voice. The text is never inspected, let alone changed."""
    floor = deps.get("tts_fn") or default_tts
    voice_tts = deps.get("voice_tts_fn") or tts_router.synthesize_voice
    requested = ann.get("voice_requested")
    rejected = ann.get("voice_reason")
    chain: list[tuple[str, str, str | None]] = []
    if ann.get("voice_id"):
        chain.append((VOICE_SOURCE_EVENT, str(ann["voice_id"]),
                      ann.get("voice_identity")))
    if ann.get("default_voice_id") and \
            ann["default_voice_id"] != ann.get("voice_id"):
        chain.append((VOICE_SOURCE_DEFAULT, str(ann["default_voice_id"]),
                      ann.get("default_voice_identity")))

    def speak(text: str) -> bytes:
        attempts: list[dict[str, Any]] = []
        for source, voice_id, identity in chain:
            refused = _recheck_announcement_voice(source, voice_id, identity)
            if refused:
                attempts.append({"source": source, "voice_id": voice_id,
                                 "unavailable": refused})
                logger.warning("announcement %s: voice %s (%s) no longer "
                               "usable (%s); trying the next voice",
                               token[:8], voice_id, source, refused)
                continue
            try:
                audio = voice_tts(text, voice_id)
            except Exception as exc:  # noqa: BLE001 - the whole point
                code = getattr(exc, "reason", None) or type(exc).__name__
                attempts.append({"source": source, "voice_id": voice_id,
                                 "unavailable": str(code)[:64]})
                logger.warning("announcement %s: voice %s (%s) unavailable "
                               "(%s); trying the next voice", token[:8],
                               voice_id, source, code)
                continue
            _record_announcement_voice(
                token, ann, requested, rejected, attempts, source, voice_id,
                tts_router.provider_for(voice_id), text, db_path)
            return audio
        try:
            audio = floor(text)
        except Exception:
            # The station voice failed too: the row fails exactly as it
            # always did (retried inside the window), but what was tried
            # first must not be lost with it.
            if attempts:
                _note(token, {"announcement_voice": {
                    "requested": requested, "authorized_voice_id":
                    ann.get("voice_id"), "resolved_voice_id": None,
                    "provider": None, "source": None, "fallback": True,
                    "reason": ANNOUNCEMENT_VOICE_FALLBACK,
                    "attempts": attempts}}, db_path)
            raise
        identity = voice_identity()
        _record_announcement_voice(
            token, ann, requested, rejected, attempts, VOICE_SOURCE_STATION,
            identity["id"], identity["provider"], text, db_path)
        return audio

    return speak


def _recheck_announcement_voice(source: str, voice_id: str,
                                identity: str | None) -> str | None:
    """The render-time authorization check for one candidate voice, as a
    reason or None. ONE contract for both selected voices: it must still
    exist, still hold the committed identity, and its CURRENT scope must
    permit announcements. The default must in addition still be the
    configured default -- the setting says which authorized voice is
    preferred, never that it is authorized. A registry that cannot answer
    refuses: never speak on a guess.

    Which reason is RECORDED when several apply: a voice that no longer
    exists, or no longer holds the committed recording, says so first.
    Deleting a custom voice also clears it as the default, and
    "the default changed" would otherwise hide the
    truer "that voice is gone". Either way nothing is spoken with it."""
    import voice_library
    try:
        reason = voice_library.check_announcement_voice(voice_id, identity)
        if reason in (voice_library.CHECK_MISSING,
                      voice_library.CHECK_IDENTITY_CHANGED):
            return reason
        if source == VOICE_SOURCE_DEFAULT:
            import settings as settings_mod
            cfg = settings_mod.announcements_config()
            if cfg.get("problem") or (cfg.get("default_voice_id") or "") != voice_id:
                return voice_library.CHECK_WITHDRAWN
        return reason
    except Exception:  # noqa: BLE001 - the registry boundary
        logger.warning("announcement voice %s could not be rechecked; not "
                       "using it", voice_id, exc_info=True)
        return "voice_check_failed"


def _record_announcement_voice(token: str, ann: dict[str, Any],
                               requested: str | None, rejected: str | None,
                               attempts: list[dict[str, Any]],
                               source: str, voice_id: str, provider: str,
                               text: str, db_path=None) -> None:
    """One structured record of the voice decision, in the row (Air Check,
    the occurrence store adopts it) and in dj_forensics (outlives the row).
    Never raises: evidence must not be able to cost an airing."""
    try:
        import voice_library
        authorized = ann.get("voice_id")
        event_refusal = next(
            (a["unavailable"] for a in attempts
             if a.get("source") == VOICE_SOURCE_EVENT
             and a.get("unavailable") in voice_library.CHECK_REASONS), None)
        if rejected:
            reason: str | None = str(rejected)
        elif event_refusal:
            # the requested voice was refused at render, not merely
            # unavailable: say which check failed
            reason = event_refusal
        elif attempts:
            reason = ANNOUNCEMENT_VOICE_FALLBACK
        else:
            reason = None
        fallback = bool(requested) and source != VOICE_SOURCE_EVENT
        record = {
            "requested": requested,
            "authorized_voice_id": authorized,
            "resolved_voice_id": voice_id or None,
            "provider": provider or None,
            "source": source,
            "fallback": fallback,
            "reason": reason if (fallback or attempts) else None,
            "attempts": attempts,
        }
        notes: dict[str, Any] = {"announcement_voice": record}
        if source != VOICE_SOURCE_STATION:
            # The row's voice facts describe THIS render, not the persona:
            # history and /dj/status answer "who said it" truthfully.
            notes["voice_id"] = voice_id
            notes["tts_provider"] = provider
        _note(token, notes, db_path)
        if fallback or attempts:
            logger.warning("announcement %s spoken by %s (%s), not the "
                           "requested %r: %s", token[:8], voice_id or "the "
                           "station voice", source, requested, reason)
        try:
            import dj_forensics as _fx
            _fx.append({
                "event": "announcement_voice",
                "at": service_clock.legacy_now().isoformat(timespec="seconds"),
                "break_id": token,
                "occurrence_id": ann.get("occurrence_id"),
                "source_calendar": ann.get("source"),
                "mode": ann.get("mode"),
                "origin": ann.get("origin"),
                "text": (text or "")[:600],
                **record,
            })
        except Exception:  # noqa: BLE001
            logger.debug("announcement voice forensics failed", exc_info=True)
    except Exception:  # noqa: BLE001
        logger.debug("announcement voice note failed", exc_info=True)


def render_request_ack(token: str, deps: dict[str, Any] | None = None,
                       now: datetime | None = None, db_path=None) -> str:
    """Render the liner and queue it only while it still owns the boundary
    it was asked for. Returns 'queued' | 'too_late' | 'render_failed' |
    'gone' | 'failed'.

    FRESHNESS is the whole point: between the button press and the finished
    audio (several seconds of LLM+TTS) the song may end. If the current queue item
    is no longer the one the listener pressed against -- or the structural
    insertion gate has closed because MA already buffered the next track --
    the liner is revoked ('too_late') rather than chased into a later
    boundary, where it would answer a request the listener has stopped
    thinking about. Every failure path leaves music untouched."""
    deps = deps or DEFAULT_DEPS or {}
    now_dt = now or service_clock.legacy_now()
    brk = get_break(token, db_path)
    if brk is None or brk["state"] != "planned":
        return "gone"          # revoked by a newer request, or already run
    req = (brk["telemetry"].get("request") or {})
    context = {
        "purpose": PURPOSE_STATION_REQUEST,
        "requested_artist": req.get("artist"),
        "current_song": req.get("song"),
        "first_new_track": req.get("first_new_track"),
        "max_words": ACK_MAX_WORDS,
        # The liner's predecessor claim is the song the button was pressed
        # against; recorded like any break's adjacency for diagnostics.
        "pre_name": req.get("song"),
        "pre_queue_item_id": req.get("queue_item_id"),
    }
    ok = render_break(token, {"session_id": brk["session_id"]}, [], None,
                      llm_fn=deps.get("llm_fn"), tts_fn=deps.get("tts_fn"),
                      transcode_fn=deps.get("transcode_fn"),
                      context=context, now=now_dt, db_path=db_path)
    if not ok:
        return "render_failed"  # already terminal 'failed'; music untouched

    import ma_queue
    read_state = deps.get("state_fn") or ma_queue.queue_state
    still_current = False
    try:
        state = read_state(brk["queue_id"])
        current = (state or {}).get("current_item") or {}
        still_current = (req.get("queue_item_id") is not None
                         and current.get("queue_item_id")
                         == req.get("queue_item_id"))
    except Exception as exc:  # unreadable queue is never a guess
        logger.warning("ack %s: queue state unreadable: %s", token[:8], exc)
    if not still_current:
        revoke(token, service_clock.legacy_now(), db_path, reason="too_late")
        return "too_late"

    result = try_insert(token, service_clock.legacy_now(), state_fn=read_state,
                        insert_fn=deps.get("insert_fn"), db_path=db_path,
                        items_fn=deps.get("items_fn"))
    if result == "queued":
        return "queued"
    if result in ("gate_closed", "not_ready", "stale"):
        # The boundary is gone (MA already buffered the next track), a newer
        # request took the row, or the liner's own assumptions stopped being
        # true while it rendered: either way this liner does not air.
        revoke(token, service_clock.legacy_now(), db_path, reason="too_late")
        return "too_late"
    return result               # 'failed' -- already terminal


def start_request_ack_render(token: str, deps: dict[str, Any] | None = None,
                             now: datetime | None = None, db_path=None) -> None:
    """Kick the liner off the request path: the button returns as soon as
    the station itself is committed, never waiting on LLM/TTS. Same daemon
    pattern as the scheduler hooks; deps['synchronous'] runs it inline for
    tests. Fenced -- a liner problem can never disturb the station change."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            render_request_ack(token, deps, now, db_path)
        except Exception:
            logger.warning("station-request ack failed (music unaffected)",
                           exc_info=True)

    if deps.get("synchronous"):
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


# --- airing observation (fed by ordinary event ingestion) ---------------------

def on_break_track_start(token: str, now: datetime | None = None,
                         db_path=None) -> bool:
    """The break's URL became the current MA item. Airtime is observed HERE,
    from the player, never inferred from HTTP fetches. The air_start stamp
    is the cadence anchor: cadence_min_minutes measures audible break start
    to audible break start."""
    now_dt = now or service_clock.legacy_now()
    ok = _transition(token, ("queued", "ready"), "airing", now_dt, db_path)
    if ok:
        _note(token, {"air_start": _iso(now_dt)}, db_path)
        brk = get_break(token, db_path) or {}
        if (brk.get("telemetry") or {}).get("purpose") == PURPOSE_NEWS_INTRO:
            # A news intro is ParlorWave speech AND part of a publisher event;
            # the news lifecycle needs to know its line is on air. Fenced:
            # a news bookkeeping problem never disturbs break airing.
            try:
                import news_schedule
                news_schedule.on_intro_start(token, now_dt, db_path)
            except Exception:
                logger.warning("news intro start hook failed", exc_info=True)
    return ok


def complete_airing(queue_id: str, now: datetime | None = None,
                    db_path=None) -> int:
    """A real music track started on this queue: any airing break is done.

    The MUSIC-BOUNDARY completion path. It is not the only one -- news
    intros are completed by news_schedule when their bulletin starts --
    so the durable record is emitted by `_transition`, which both paths
    share, rather than here. The render-time record (event `aired`,
    lifecycle `rendered`) cannot answer "was it heard": a break is
    revoked after rendering often enough to matter -- a fifth or so of
    rendered breaks may never play -- so analytics must join on
    completion, not on the render row. The row state remains the
    authority; the stream is a durable projection of it that outlives the
    reaped `dj_breaks` row."""
    now_dt = now or service_clock.legacy_now()
    c = _connect(db_path)
    try:
        tokens = [r[0] for r in c.execute(
            "SELECT token FROM dj_breaks WHERE queue_id = ? "
            "AND state = 'airing'", (queue_id,)).fetchall()]
    finally:
        c.close()
    # ONE transition per token rather than one bulk UPDATE, so completion
    # has a single durable emission point (_transition) instead of two
    # that must be kept in step. Each is still atomic and still guarded by
    # rowcount==1, which is strictly stronger than the bulk form: a racing
    # caller loses cleanly and emits nothing. The return value is
    # unchanged -- how many breaks this call actually completed.
    return sum(1 for tok in tokens
               if _transition(tok, ("airing",), STATE_COMPLETED,
                              now_dt, db_path))


# --- maintenance --------------------------------------------------------------

def recover_orphaned_renders(now: datetime | None = None,
                             db_path=None) -> list[dict[str, Any]]:
    """Revoke every 'rendering' row that no living render can finish: owned
    by another process, or owned by nobody and last touched before this
    process started (a build from before render owners existed). Run at
    boot, before anything can plan, so a restart mid-render costs one break
    and never blocks the DJ until the stale-break TTL.

    PRE-HANDOFF BY CONSTRUCTION: 'rendering' audio is never servable and
    nothing inserts from 'rendering', so no revoked row ever reached Music
    Assistant; its token answers the retired-silence tombstone from here
    on. 'revoked', not 'failed': this is a deliberate withdrawal, and a
    failure would hold the planner in its retry cooldown. Everything past
    rendering is left alone -- 'ready' may already be with MA (it probes at
    enqueue), 'queued'/'airing' are MA's, and the gate, the observer and
    revoke_stale own those. A render THIS process owns is never touched.
    Returns what it revoked."""
    now_dt = now or service_clock.legacy_now()
    started = _iso(PROCESS_STARTED_AT)
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT token, queue_id, updated_at, telemetry FROM dj_breaks "
            "WHERE state = 'rendering'").fetchall()
    finally:
        c.close()
    revoked: list[dict[str, Any]] = []
    for token, queue_id, updated_at, raw in rows:
        try:
            tel = json.loads(raw or "{}")
        except (TypeError, ValueError):
            tel = {}
        if not isinstance(tel, dict):
            tel = {}
        owner = tel.get("render_owner")
        if owner == PROCESS_RENDER_OWNER:
            continue                      # in flight here, right now
        if owner is None and str(updated_at or "") >= started:
            continue                      # ownerless but touched since boot
        if _transition(token, ("rendering",), "revoked", now_dt, db_path,
                       error=ORPHANED_RENDER):
            revoked.append({"token": token, "queue_id": queue_id,
                            "owner": owner, "since": updated_at,
                            "purpose": break_purpose(tel)})
            logger.warning(
                "dj break %s: render orphaned by a previous process (owner "
                "%s, %s, rendering since %s) -- revoked; the DJ plans again "
                "at once", token[:8], owner or "none", break_purpose(tel),
                updated_at)
    return revoked


def revoke_stale(now: datetime | None = None, db_path=None) -> int:
    """ready/queued breaks older than the TTL become 'missed' (their audio
    answers the retired-silence tombstone from that moment). Stale
    planned/rendering rows fail too -- a restart mid-preparation must not
    leave immortal intents. This is the age-based BACKSTOP; the prompt
    path for a queued break the queue moved past without airing is
    reconcile_missed_queued at the next music boundary."""
    now_dt = now or service_clock.legacy_now()
    cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(minutes=READY_TTL_MINUTES))
    c = _connect(db_path)
    try:
        # Target-aware exception, MANUAL PLANNED ONLY: "DJ after this song"
        # on a track longer than the generic TTL must survive until its
        # grounded boundary. The intent stays valid strictly while its own
        # target_air_at is still in the future; the moment the boundary
        # passes (aired, retargeted, or the pin died) the generic age rule
        # applies again -- nothing is immortal. Scheduled breaks and every
        # rendering/ready/queued row keep the existing bound unchanged.
        cur = c.execute(
            "UPDATE dj_breaks SET state = CASE WHEN state IN ('ready','queued') "
            "THEN 'missed' ELSE 'failed' END, updated_at = ? "
            "WHERE state IN ('planned','rendering','ready','queued') "
            "AND updated_at < ? "
            "AND NOT (state = 'planned' "
            "         AND json_extract(telemetry, '$.purpose') = ? "
            "         AND json_extract(telemetry, '$.target_air_at') > ?)",
            (_iso(now_dt), cutoff, PURPOSE_MANUAL,
             service_clock.cutoff_destructive(now_dt, timedelta(0))))
        c.commit()
        return cur.rowcount
    finally:
        c.close()


def retire_expired(now: datetime | None = None, db_path=None) -> dict[str, int]:
    """Bounded retention for terminal DJ breaks.

    Terminal rows (completed/missed/revoked/failed) past their retention age
    -- and, cap pass, the oldest terminal rows beyond BREAK_ROW_CAP -- are
    retired: audio file FIRST, row SECOND. The row is the durable to-do
    marker: a crash between the two leaves a terminal row whose unlink
    retries idempotently on the next sweep, whereas the reverse order could
    strand an invisible orphan MP3. Non-terminal rows are never touched at
    any age (revoke_stale owns their staleness). Token-shaped orphan MP3s
    with no row (never servable: the audio gate 404s unknown tokens) are
    collected once older than the completed window; anything not
    token-shaped is left in place. A failed unlink skips that row and never
    aborts the sweep."""
    now_dt = now or datetime.now()
    stats = {"rows": 0, "files": 0, "orphans": 0, "retired_facts": 0}
    completed_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(hours=RETAIN_COMPLETED_H))
    terminal_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(hours=RETAIN_TERMINAL_H))
    terminal_ph = ",".join("?" for _ in TERMINAL_STATES)
    c = _connect(db_path)
    try:
        expired = c.execute(
            f"SELECT token FROM dj_breaks WHERE state IN ({terminal_ph}) "
            "AND ((state = 'completed' AND updated_at < ?) OR "
            "(state != 'completed' AND updated_at < ?)) ORDER BY updated_at",
            (*TERMINAL_STATES, completed_cutoff, terminal_cutoff)).fetchall()
        victims = [t for (t,) in expired]
        newest_first = c.execute(
            f"SELECT token FROM dj_breaks WHERE state IN ({terminal_ph}) "
            "ORDER BY updated_at DESC", TERMINAL_STATES).fetchall()
        for (t,) in newest_first[BREAK_ROW_CAP:]:
            if t not in victims:
                victims.append(t)
        import program_tokens
        for token in victims:
            path = audio_dir() / f"{token}.mp3"
            try:
                if path.exists():
                    path.unlink()
                    stats["files"] += 1
            except OSError:
                logger.warning("break audio unlink failed, will retry next "
                               "sweep: %s...", token[:8])
                continue
            # THE FACT BEFORE THE FORGETTING, on this same connection and
            # inside this same transaction. The row is what makes a token
            # KNOWN, and a known token must never become transport-fatal
            # (program_audio) -- so the moment the row stops existing, the
            # retirement has to already be recorded. Doing it in two
            # transactions would leave a crash window in which a token is
            # neither live nor known, which is precisely the 404 this
            # closes.
            program_tokens.record(c, [token], program_tokens.KIND_DJ, now_dt)
            cur = c.execute(
                f"DELETE FROM dj_breaks WHERE token = ? AND state IN "
                f"({terminal_ph})", (token, *TERMINAL_STATES))
            stats["rows"] += cur.rowcount
            stats["retired_facts"] = stats.get("retired_facts", 0) + 1
        # Skip markers stranded by a station moving to another queue (their
        # queue_id never matches a planner again) are bounded the same way
        # terminal rows are: gone once older than the terminal window.
        c.execute(
            "DELETE FROM meta WHERE key LIKE 'dj_skip:%' "
            "AND json_extract(value, '$.suppressed_target') < ?",
            (terminal_cutoff,))
        c.commit()
        live = {t for (t,) in c.execute("SELECT token FROM dj_breaks")}
    finally:
        c.close()
    orphan_cutoff = (service_clock.instant(now_dt)
                     - timedelta(hours=RETAIN_COMPLETED_H)).timestamp()
    adir = audio_dir()
    if adir.is_dir():
        for path in adir.glob("*.mp3"):
            if not _TOKEN_RE.match(path.stem):
                logger.warning("unrecognized file in broadcast_audio left "
                               "in place: %s", path.name)
                continue
            if path.stem in live:
                continue
            try:
                if path.stat().st_mtime < orphan_cutoff:
                    path.unlink()
                    stats["orphans"] += 1
            except OSError:
                pass
    return stats


def revoke(token: str, now: datetime | None = None, db_path=None,
           reason: str | None = None) -> bool:
    fields = {"error": reason} if reason else {}
    return _transition(token, NON_TERMINAL_STATES, "revoked",
                       now or service_clock.legacy_now(), db_path, **fields)


# --- manual break control (Force / Cancel) ------------------------------------
# Force = "DJ after this song": a first-class ordinary intent structurally
# pinned to the queue item that was current when the listener asked, riding
# the unchanged render/gate/insert machinery. Cancel = suppress exactly one
# ordinary cadence opportunity, whether or not an intent row exists yet.
# Neither ever takes a transport action or fakes an air anchor.

# Telemetry keys that survive a manual retarget. Everything else (frozen
# adjacency, warm/render/queue stamps) was true of the OLD boundary and is
# discarded with the stale render.
_MANUAL_KEEP_KEYS = ("intent_created", "purpose", "origin",
                     "skipped_opportunity")


def _is_music_item(item: dict[str, Any] | None) -> bool:
    """False for our own finite programming (DJ break / news audio URLs);
    True for ordinary music. Deterministic: the same URL identity the audio
    routes serve."""
    uri = str((item or {}).get("uri") or "")
    return "/broadcast/audio/" not in uri and "/news/audio/" not in uri


def _force_target(state: dict[str, Any],
                  now_dt: datetime) -> tuple[str, str, datetime] | None:
    """(queue_item_id, item_name, target datetime) meaning 'the natural end
    of the current item', or None when that boundary cannot be grounded.
    Fail-closed by contract: no duration/elapsed evidence, a stale MA
    checkpoint, a non-playing player, or a non-music current item all refuse
    rather than guess. Unlike estimated_air_at there is deliberately no
    maximum-horizon cap: a manual request against a long track is scheduled
    for that track's real boundary, however far out."""
    if (state or {}).get("state") != "playing":
        return None
    item = state.get("current_item") or {}
    qid = item.get("queue_item_id")
    if not qid or not _is_music_item(item):
        return None
    updated = state.get("elapsed_time_last_updated")
    if updated is not None:
        try:
            age = time.time() - float(updated)
        except (TypeError, ValueError):
            return None
        if age < 0 or age > TICK_LIVE_S:
            return None
    remaining = _remaining_s(state)
    if remaining is None:
        return None
    return (str(qid), str(item.get("name") or ""),
            now_dt + timedelta(seconds=remaining))


def _terminate_manual(token: str, reason: str, now_dt: datetime,
                      db_path=None) -> bool:
    """A manual request that can no longer be honored ends truthfully as
    'missed' (its audio answers the retired-silence tombstone from that
    moment). Never replanned."""
    return _transition(token, ("planned", "rendering", "ready"), "missed",
                       now_dt, db_path, error=reason)


def _reset_manual_row(token: str, new_tel: dict[str, Any],
                      now_dt: datetime, db_path=None) -> bool:
    """Return a manual intent to a clean 'planned' state with rebuilt
    telemetry, discarding ALL rendered material (text, audio, adjacency,
    stamps) in one conditional UPDATE. Nothing rendered against an old
    boundary may survive into a new one."""
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE dj_breaks SET state = 'planned', spoken_text = NULL, "
            "audio_path = NULL, duration_s = NULL, error = NULL, "
            "updated_at = ?, telemetry = ? "
            "WHERE token = ? AND state IN ('planned', 'ready')",
            (_iso(now_dt), json.dumps(new_tel), token))
        c.commit()
        won = cur.rowcount == 1
    finally:
        c.close()
    if won:
        try:
            path = audio_dir() / f"{token}.mp3"
            if path.exists():
                path.unlink()
        except OSError:
            logger.warning("stale manual audio unlink failed for %s...; "
                           "retention will collect it", token[:8])
    return won


def _manual_pin_check(brk: dict[str, Any], state: dict[str, Any],
                      now_dt: datetime, db_path=None) -> str | None:
    """Enforce the structural pin of a manual intent against the live queue.

    Pin holds (or row is not an actionable manual intent): None -- the
    ordinary flow proceeds. Pin broken: one bounded re-pin to the new
    current MUSIC item ('manual_retargeted', retarget_count persisted, all
    stale render material discarded), else terminal ('manual_terminated'):
    a second invalidation, a non-music current item, or an ungroundable new
    boundary never chases further."""
    tel = brk.get("telemetry") or {}
    if break_purpose(tel) != PURPOSE_MANUAL:
        return None
    if brk["state"] not in ("planned", "ready"):
        return None
    if (state or {}).get("state") != "playing":
        # A paused/idle queue is not evidence the pin broke -- the pinned
        # item may resume. Waiting is bounded: once the target boundary
        # passes, the planned-TTL exception lapses and the generic TTL
        # retires the intent.
        return None
    pinned = tel.get("intended_boundary_queue_item_id")
    cur_id = (state.get("current_item") or {}).get("queue_item_id")
    if not pinned or not cur_id or str(cur_id) == str(pinned):
        return None
    token = brk["token"]
    retargets = int(tel.get("retarget_count") or 0)
    if retargets >= MANUAL_RETARGET_MAX:
        if _terminate_manual(token, "retarget_exhausted", now_dt, db_path):
            return "manual_terminated"
        return None  # raced past us; let the ordinary flow see the new state
    if not _is_music_item(state.get("current_item")):
        # Never pin DJ speech or a news bulletin as "this song" -- and
        # waiting behind our own programming would need new machinery.
        # Terminate truthfully instead (smallest deterministic behavior).
        if _terminate_manual(token, "retarget_nonmusic", now_dt, db_path):
            return "manual_terminated"
        return None
    grounded = _force_target(state, now_dt)
    if grounded is None:
        if _terminate_manual(token, "retarget_unavailable", now_dt, db_path):
            return "manual_terminated"
        return None
    new_id, new_name, new_target = grounded
    new_tel = {k: tel[k] for k in _MANUAL_KEEP_KEYS if k in tel}
    new_tel.update({
        "intended_boundary_queue_item_id": new_id,
        "intended_boundary_name": new_name,
        "target_air_at": _iso(new_target),
        "retarget_count": retargets + 1,
        "retargeted_at": _iso(now_dt),
        "retargeted_from": pinned,
    })
    if _reset_manual_row(token, new_tel, now_dt, db_path):
        return "manual_retargeted"
    return None


def _session_event(session_id: str, kind: str, payload: dict[str, Any],
                   now_dt: datetime, db_path=None) -> None:
    """Best-effort truthful history: a human's manual action is observable
    in session_events without ever touching break lifecycle or anchors.
    Written only when a real session exists."""
    if not session_id:
        return
    try:
        c = _connect(db_path)
        try:
            c.execute(
                "INSERT INTO session_events (session_id, at, kind, payload) "
                "VALUES (?, ?, ?, ?)",
                (session_id, _iso(now_dt), kind, json.dumps(payload)))
            c.commit()
        finally:
            c.close()
    except Exception:
        logger.debug("session event %s not recorded", kind, exc_info=True)


def _quiet_slot(token: str, queue_id: str, brk: dict[str, Any],
                session: dict[str, Any], extras: dict[str, Any],
                now_dt: datetime, db_path=None,
                calendar_event: dict[str, Any] | None = None) -> bool:
    """LOW chattiness: skip ONE purely discretionary cadence slot.

    INTERNAL ONLY since the persona/airtime decoupling. Nothing
    a customer can select resolves LOW (dj_persona.effective_chattiness
    answers PRODUCTION_CHATTINESS for every personality and every stored
    value), so `chat_policy["quiet_slots"]` is False on every production
    render and this never runs there. It is kept, not deleted, because it
    is still the machinery a test or an offline bake-off drives by naming
    the level -- and the known LOW repeated-quiet-slot defect is left
    where it is rather than widened into this change.

    Applies only to a planned, ordinary, SCHEDULED intent (never a manual
    Force, a liner or the news intro) that is not itself the make-good for
    an earlier skip, and only when NO contextual topic was selected for
    it -- by the deterministic editor (upcoming_event / news_headline) or,
    on the Smart DJ path, by the calendar packet, which is where a
    qualified event now lives instead. Checking only the editor's keys
    would silence exactly the slot a one-hour-out event had just made
    worth keeping. Either way the test is the same: would the DJ be
    talking merely because the clock said so?

    It rides the Cancel-Next-Break machinery
    exactly: the row is revoked and the dj_skip marker written in ONE
    transaction, so plan_break's next intent targets one cadence after the
    suppressed slot and carries skipped_opportunity, which exempts it from
    a second skip. Cadence therefore remains the floor between audible
    breaks and at most every other slot goes quiet. No transport action;
    no air anchor is faked."""
    tel = brk.get("telemetry") or {}
    if brk.get("state") != "planned":
        return False
    if break_purpose(tel) != PURPOSE_SCHEDULED:
        return False
    if tel.get("skipped_opportunity"):
        return False
    if calendar_event or any(extras.get(k) for k in
                             ("upcoming_event", "news_headline")):
        return False
    target_iso = tel.get("target_air_at") or _iso(now_dt)
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        cur = c.execute(
            "UPDATE dj_breaks SET state = 'revoked', error = 'quiet_slot', "
            "updated_at = ? WHERE token = ? AND state = 'planned'",
            (_iso(now_dt), token))
        if cur.rowcount != 1:
            c.execute("ROLLBACK")
            return False
        marker = {"suppressed_target": target_iso,
                  "created_at": _iso(now_dt), "source": "chattiness_low",
                  "revoked_token": token}
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                  (_skip_key(queue_id), json.dumps(marker)))
        c.commit()
    finally:
        c.close()
    _session_event(session.get("session_id") or "", "dj_break_quiet",
                   {"token": token, "suppressed_target": target_iso,
                    "source": "chattiness_low"}, now_dt, db_path)
    logger.info("dj quiet slot: chattiness low, nothing worth saying; "
                "skipped %s (next break targets one cadence later)",
                target_iso)
    return True


def _close_for_speech(token: str, queue_id: str, brk: dict[str, Any],
                      session: dict[str, Any], now_dt: datetime,
                      db_path=None) -> bool:
    """DJ speech is off and an ORDINARY opportunity is here: close it with
    the reason dj_speech_disabled, render nothing, and move cadence on.

    The row (planned, or ready if it was rendered before the switch
    flipped) is revoked and the dj_skip marker is written in ONE
    transaction -- the Cancel-Next-Break machinery exactly, as _quiet_slot
    also uses it -- so plan_break's next intent targets one cadence after
    this slot instead of re-planning an already-due intent on every tick.
    One ledger row per cadence slot, each saying why it was silent.

    A break already handed to Music Assistant (queued / airing) is never
    candidate here and plays out, exactly as Cancel answers too_late for
    it: this switch changes what is written next, never the queue.

    Nothing else is touched. In the ordinary case the slot closes BEFORE
    any render: no context was gathered, so no service presence
    (clock/weather) was marked aired and no calendar band or fact novelty
    was spent -- every obligation this slot would have carried stays owed
    to the next spoken break. A row closed AFTER it rendered (the switch
    flipped mid-render, or while it sat ready) keeps whatever its render
    already recorded, exactly like a stale-context revocation after render:
    rendered, never heard. No failure, planner decline or insert trace is
    recorded by the closing itself."""
    tel = brk.get("telemetry") or {}
    target_iso = tel.get("target_air_at") or _iso(now_dt)
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        cur = c.execute(
            "UPDATE dj_breaks SET state = 'revoked', error = ?, "
            "updated_at = ? WHERE token = ? AND state IN ('planned', 'ready')",
            (SPEECH_DISABLED, _iso(now_dt), token))
        if cur.rowcount != 1:
            c.execute("ROLLBACK")
            return False
        marker = {"suppressed_target": target_iso,
                  "created_at": _iso(now_dt), "source": SPEECH_DISABLED,
                  "revoked_token": token}
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                  (_skip_key(queue_id), json.dumps(marker)))
        c.commit()
    finally:
        c.close()
    purpose = break_purpose(tel)
    _session_event(session.get("session_id") or "", "dj_break_speech_disabled",
                   {"token": token, "suppressed_target": target_iso,
                    "purpose": purpose, "state": brk.get("state")},
                   now_dt, db_path)
    import dj_forensics as _fx
    _fx.suppressed(token, queue_id, SPEECH_DISABLED, purpose,
                   brk.get("state"), target_iso, now_dt)
    logger.info("dj speech off: %s opportunity %s closed (%s); next "
                "targets one cadence later", purpose, target_iso,
                SPEECH_DISABLED)
    return True


def cancel_next_break(queue_id: str, session: dict[str, Any] | None = None,
                      now: datetime | None = None, db_path=None) -> dict[str, Any]:
    """Cancel Next Break: suppress exactly ONE ordinary cadence opportunity.

    Existing ordinary intent in planned/rendering/ready: conditional revoke
    AND the dj_skip marker land in ONE transaction -- there is no crash
    window where the row is revoked but the opportunity would be replanned.
    queued/airing: too_late, truthfully (a 410 cannot un-fetch audio MA may
    already hold). No intent: the marker alone suppresses the upcoming
    opportunity, derived from the same anchor arithmetic plan_break uses.
    Event liners and news are invisible here. Idempotent while a marker is
    pending."""
    now_dt = now or service_clock.legacy_now()
    session = session or {}
    # Pre-compute the no-row suppression target OUTSIDE the transaction
    # (read-only helpers open their own connections). If a row exists the
    # row's own target wins inside the transaction.
    anchor = max(filter(None, [session.get("started_at"),
                               last_air_anchor(queue_id, db_path)]),
                 default=None)
    try:
        anchor_dt = datetime.fromisoformat(anchor) if anchor else now_dt
    except (TypeError, ValueError):
        anchor_dt = now_dt
    derived_target = max(now_dt, service_clock.legacy_shift(
        anchor_dt, timedelta(minutes=cadence_minutes())))
    nt_ph = ",".join("?" for _ in NON_TERMINAL_STATES)
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        rows = c.execute(
            f"SELECT token, state, telemetry FROM dj_breaks WHERE "
            f"queue_id = ? AND state IN ({nt_ph}) ORDER BY created_at DESC",
            (queue_id, *NON_TERMINAL_STATES)).fetchall()
        ordinary = []
        for t, s, tel_raw in rows:
            try:
                tel = json.loads(tel_raw or "{}")
            except (TypeError, ValueError):
                tel = {}
            if is_ordinary_break(tel):
                ordinary.append((t, s, tel))
        for t, s, _tel in ordinary:
            if s in ("queued", "airing"):
                c.execute("ROLLBACK")
                return {"ok": False, "reason": "too_late",
                        "state": _STATUS_LABELS.get(s, s)}
        marker_row = c.execute("SELECT value FROM meta WHERE key = ?",
                               (_skip_key(queue_id),)).fetchone()
        cancellable = next((r for r in ordinary
                            if r[1] in ("planned", "rendering", "ready")),
                           None)
        if cancellable is not None:
            token, _s, tel = cancellable
            cur = c.execute(
                "UPDATE dj_breaks SET state = 'revoked', "
                "error = 'manual_cancel', updated_at = ? WHERE token = ? "
                "AND state IN ('planned', 'rendering', 'ready')",
                (_iso(now_dt), token))
            if cur.rowcount != 1:
                c.execute("ROLLBACK")  # escaped into MA between read and write
                return {"ok": False, "reason": "too_late", "state": "up_next"}
            target_iso = tel.get("target_air_at") or _iso(derived_target)
            marker = {"suppressed_target": target_iso,
                      "created_at": _iso(now_dt), "source": "manual_cancel",
                      "revoked_token": token}
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                      (_skip_key(queue_id), json.dumps(marker)))
            c.commit()
            result = {"ok": True, "cancelled": token,
                      "suppressed_target": target_iso}
        elif marker_row is not None:
            c.execute("ROLLBACK")  # nothing to change: idempotent
            try:
                pending = json.loads(marker_row[0]).get("suppressed_target")
            except (TypeError, ValueError, AttributeError):
                pending = None
            return {"ok": True, "cancelled": "already_suppressed",
                    "suppressed_target": pending}
        else:
            target_iso = _iso(derived_target)
            marker = {"suppressed_target": target_iso,
                      "created_at": _iso(now_dt), "source": "manual_cancel",
                      "revoked_token": None}
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                      (_skip_key(queue_id), json.dumps(marker)))
            c.commit()
            result = {"ok": True, "cancelled": "next_opportunity",
                      "suppressed_target": target_iso}
    finally:
        c.close()
    _session_event(session.get("session_id") or "", "dj_break_cancelled",
                   {"suppressed_target": result.get("suppressed_target"),
                    "revoked_token": result.get("cancelled")
                    if result.get("cancelled") not in
                    ("next_opportunity", "already_suppressed") else None},
                   now_dt, db_path)
    return result


#: A planned ordinary intent whose target falls within this many seconds
#: of now is DUE at the boundary a standalone announcement is about to take:
#: the announcement replaces it rather than
#: shifting it one song later.
ANNOUNCEMENT_REPLACES_WITHIN_S = float(
    os.environ.get("DJ_ANNOUNCEMENT_REPLACES_WITHIN_S", "180"))


def yield_opportunity(queue_id: str, token: str, now: datetime | None = None,
                      reason: str = "announcement_replaced",
                      db_path=None) -> dict[str, Any] | None:
    """A standalone announcement REPLACES this due ordinary opportunity.

    The PLANNED intent is revoked and the dj_skip marker written in ONE
    transaction -- Cancel Next Break's own mechanism -- so the replanned
    intent targets one cadence later instead of airing one song after the
    announcement. Nothing is marked as said: a clock, weather or calendar
    item that intent would have owed stays owed for the next break. Only a
    planned intent (nothing rendered, nothing committed) is ever taken; a
    pending Cancel Next Break marker is left as it is. Returns the marker,
    or None when nothing was taken."""
    now_dt = now or service_clock.legacy_now()
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT state, telemetry FROM dj_breaks WHERE "
                        "token = ? AND queue_id = ?", (token, queue_id)
                        ).fetchone()
        if row is None or row[0] != "planned":
            c.execute("ROLLBACK")
            return None
        try:
            tel = json.loads(row[1] or "{}")
        except (TypeError, ValueError):
            tel = {}
        if break_purpose(tel) != PURPOSE_SCHEDULED or not tel.get(
                "target_air_at"):
            c.execute("ROLLBACK")
            return None
        if c.execute("SELECT 1 FROM meta WHERE key = ?",
                     (_skip_key(queue_id),)).fetchone():
            c.execute("ROLLBACK")
            return None
        c.execute("UPDATE dj_breaks SET state = 'revoked', error = ?, "
                  "updated_at = ? WHERE token = ? AND state = 'planned'",
                  (reason, _iso(now_dt), token))
        marker = {"suppressed_target": tel["target_air_at"],
                  "created_at": _iso(now_dt), "source": reason,
                  "revoked_token": token}
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                  (_skip_key(queue_id), json.dumps(marker)))
        c.commit()
        return marker
    finally:
        c.close()


def yield_upcoming(queue_id: str, session: dict[str, Any],
                   now: datetime | None = None,
                   reason: str = "announcement_replaced",
                   db_path=None) -> dict[str, Any] | None:
    """yield_opportunity for an opportunity that has NO intent row yet.

    At a music boundary the announcement pass runs before the DJ's own pass
    (on_music_track_start), so the next ordinary intent is often not planned
    yet when a standalone announcement takes the boundary -- and would then
    be planned, already due, one song after it.
    When the next opportunity (the same anchor + cadence arithmetic
    plan_break uses) falls within ANNOUNCEMENT_REPLACES_WITHIN_S, the skip
    marker alone consumes it, exactly as Cancel Next Break does with no row.
    Nothing is written when any speech is in flight or a marker is pending."""
    now_dt = now or service_clock.legacy_now()
    target = next_ordinary_opportunity(queue_id, session or {}, now_dt,
                                       db_path)
    if target is None or target > now_dt + timedelta(
            seconds=ANNOUNCEMENT_REPLACES_WITHIN_S):
        return None
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        live = c.execute(
            "SELECT COUNT(*) FROM dj_breaks WHERE queue_id = ? AND state IN "
            f"({','.join('?' for _ in NON_TERMINAL_STATES)}) AND "
            "COALESCE(json_extract(telemetry, '$.purpose'), 'scheduled') "
            "IN ('scheduled', 'manual')",
            (queue_id, *NON_TERMINAL_STATES)).fetchone()[0]
        if live or c.execute("SELECT 1 FROM meta WHERE key = ?",
                             (_skip_key(queue_id),)).fetchone():
            c.execute("ROLLBACK")
            return None
        marker = {"suppressed_target": _iso(target),
                  "created_at": _iso(now_dt), "source": reason,
                  "revoked_token": None}
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                  (_skip_key(queue_id), json.dumps(marker)))
        c.commit()
        return marker
    finally:
        c.close()


def force_break(queue_id: str, state: dict[str, Any],
                session: dict[str, Any] | None = None,
                now: datetime | None = None, db_path=None) -> dict[str, Any]:
    """DJ AFTER THIS SONG: create a manual intent structurally pinned to the
    item that is current RIGHT NOW, targeting that item's natural ending
    boundary. One transaction clears any pending skip marker, supersedes an
    ordinary planned/rendering intent, and creates the manual row -- an
    internal failure rolls back all three, leaving the prior world intact.

    Refusals (no intent created, nothing changed): boundary_unavailable
    (the current item's end cannot be grounded -- fail closed, never
    schedule a guess), break_imminent (ordinary speech already
    ready/queued/airing), other_speech_in_flight (liner or news intro
    holds the single in-flight slot), news_in_flight / quiet_guard (the
    news engine owns the queue right now)."""
    now_dt = now or service_clock.legacy_now()
    session = session or {}
    # A manual break is ordinary DJ speech: with DJ speech switched off it
    # is refused up front, honestly, rather than planned only to be closed
    # at its boundary. Nothing is changed -- a pending skip stays pending.
    if not dj_speech_enabled():
        return {"ok": False, "reason": SPEECH_DISABLED}
    try:
        import news_schedule
        blocked = news_schedule.ordinary_dj_block_reason(session, queue_id,
                                                         db_path=db_path,
                                                         now=now_dt)
    except Exception:
        blocked = None
    if blocked:
        return {"ok": False, "reason": blocked}
    grounded = _force_target(state, now_dt)
    if grounded is None:
        return {"ok": False, "reason": "boundary_unavailable"}
    pin_id, pin_name, target_dt = grounded
    token = _secrets.token_hex(16)
    nt_ph = ",".join("?" for _ in NON_TERMINAL_STATES)
    c = _connect(db_path)
    try:
        c.execute("BEGIN IMMEDIATE")
        rows = c.execute(
            f"SELECT token, state, telemetry FROM dj_breaks WHERE "
            f"queue_id = ? AND state IN ({nt_ph}) ORDER BY created_at DESC",
            (queue_id, *NON_TERMINAL_STATES)).fetchall()
        to_supersede = []
        for t, s, tel_raw in rows:
            try:
                tel = json.loads(tel_raw or "{}")
            except (TypeError, ValueError):
                tel = {}
            if not is_ordinary_break(tel):
                c.execute("ROLLBACK")
                return {"ok": False, "reason": "other_speech_in_flight",
                        "purpose": break_purpose(tel)}
            if s in ("ready", "queued", "airing"):
                c.execute("ROLLBACK")
                return {"ok": False, "reason": "break_imminent",
                        "state": _STATUS_LABELS.get(s, s)}
            to_supersede.append(t)
        for t in to_supersede:
            cur = c.execute(
                "UPDATE dj_breaks SET state = 'revoked', "
                "error = 'superseded_by_manual', updated_at = ? "
                "WHERE token = ? AND state IN ('planned', 'rendering')",
                (_iso(now_dt), t))
            if cur.rowcount != 1:  # raced into ready+: refuse, change nothing
                c.execute("ROLLBACK")
                return {"ok": False, "reason": "break_imminent",
                        "state": "ready"}
        c.execute("DELETE FROM meta WHERE key = ?", (_skip_key(queue_id),))
        tel = {"intent_created": _iso(now_dt), "purpose": PURPOSE_MANUAL,
               "origin": "manual_force",
               "target_air_at": _iso(target_dt),
               "intended_boundary_queue_item_id": pin_id,
               "intended_boundary_name": pin_name,
               "retarget_count": 0}
        c.execute(
            "INSERT INTO dj_breaks (token, queue_id, session_id, state, "
            "created_at, updated_at, telemetry) VALUES (?, ?, ?, 'planned', ?, ?, ?)",
            (token, queue_id, session.get("session_id") or "", _iso(now_dt),
             _iso(now_dt), json.dumps(tel)))
        c.commit()
    finally:
        c.close()
    _session_event(session.get("session_id") or "", "dj_break_forced",
                   {"token": token, "after_item": pin_id,
                    "estimated_air_at": _iso(target_dt)}, now_dt, db_path)
    return {"ok": True, "forced": token, "after_item": pin_id,
            "estimated_air_at": _iso(target_dt)}


# --- audio endpoint gate ------------------------------------------------------

def audio_available(token: str, db_path=None) -> tuple[int, Path | None]:
    """(http_status, path) for the finite-audio endpoint. Tokens grant access
    ONLY to their own rendered break file, whose path was written by
    render_break -- never to arbitrary filesystem content.

    Transport safety (program_audio): a KNOWN token that is no longer live
    (terminal state, or its file already retired) answers 200 with the
    retired-silence tombstone instead of 410. MA queue history keeps aired
    break items addressable forever, and a history replay -- listener-chosen
    or an MA flow-restart rewind -- must meet an ordinary short track, not
    a 410 storm (see program_audio). Unknown or malformed tokens stay an honest
    404: resilience is for our own history, never for probes."""
    if not _TOKEN_RE.match(token or ""):
        return 404, None
    import program_audio
    brk = get_break(token, db_path)
    if brk is None:
        # NO ROW IS NOT THE SAME AS NEVER EXISTED. Terminal rows are reaped
        # after a day; Music Assistant queue items are addressable forever,
        # and a restart can put an aired one back in front of the listener
        # (such tokens would otherwise answer 404). The ledger
        # remembers the retirement after the row is gone, so the token keeps
        # the tombstone it is entitled to. A token in neither place is
        # genuinely unknown and still gets an honest 404 -- resilience is for
        # our own history, never for probes.
        import program_tokens
        if program_tokens.is_retired(token, db_path):
            stone = program_audio.tombstone()
            return (200, stone) if stone else (410, None)
        return 404, None
    if brk["state"] not in SERVABLE_STATES:
        stone = program_audio.tombstone()
        return (200, stone) if stone else (410, None)
    path = Path(brk["audio_path"] or "")
    expected = audio_dir() / f"{token}.mp3"
    if not brk["audio_path"] or path != expected or not path.exists():
        stone = program_audio.tombstone()
        return (200, stone) if stone else (410, None)
    return 200, path


# --- read-only status (GET /dj/status; Music Brain is the state authority) ----

# Internal lifecycle -> user-facing status vocabulary. The dashboard never
# sees claim/gate/stale words; the internal lifecycle itself is unchanged.
_STATUS_LABELS = {
    "planned": "waiting",       # intent exists; render window not open yet
    "rendering": "preparing",   # LLM/TTS/audio being produced
    "ready": "ready",           # audio rendered, awaiting insertion
    "queued": "up_next",        # inserted as the NEXT MA item
    "airing": "on_air",         # /broadcast/audio/ is the current item
}


def _public_ref(token: str | None) -> str | None:
    import now_playing
    return now_playing.program_ref("dj_break", token)


def _break_detail(tel: dict[str, Any], spoken_text: str) -> dict[str, Any]:
    """The describable half of a break -- what it said, in whose voice, about
    what. Shared by `current_break` and `last_break` so the break ON AIR and
    the break that just finished are described in exactly the same words.

    They were not, and that was the bug: `current_break` carried only its
    token and lifecycle state, so a panel wanting a script had nothing to
    show but `last_break` -- and displayed the PREVIOUS break's text while a
    new one was audibly playing."""
    return {"air_start": tel.get("air_start"),
            "spoken_text": spoken_text,
            "topics": tel.get("context_topics") or [],
            "kind": tel.get("purpose") or PURPOSE_SCHEDULED,
            "personality": dj_persona.canonical_profile_id(
                tel.get("persona_profile")),
            "chattiness": tel.get("chattiness"),
            # the logical voice first; rows written before it was recorded
            # still answer with their asset
            "voice": (tel.get("voice_id") or tel.get("tts_voice")),
            "voice_provider": tel.get("tts_provider"),
            "writer": tel.get("writer"),
            # Smart DJ: which kind the planner chose (and its topic key)
            # for a break it wrote; null for legacy / deterministic copy.
            "planner": ({"action": (tel.get("planner") or {}).get("action"),
                         "topic_key": (tel.get("planner") or {}).get("topic_key")}
                        if tel.get("planner") else None)}


def _reaction_for_path(path: str | None) -> dict[str, Any] | None:
    """Catalog reaction state for one resolved track path. Read-only; never
    creates records; None when the track is not in the catalog."""
    if not path:
        return None
    try:
        import musicdb
        c = sqlite3.connect(str(musicdb.DB))
        try:
            row = c.execute(
                "SELECT thumbs_up_count, thumbs_down_count, personal_score "
                "FROM tracks WHERE path = ?", (path,)).fetchone()
        finally:
            c.close()
    except Exception:
        return None
    if row is None:
        return None
    return {"thumbs_up": (row[0] or 0) > 0,
            "thumbs_down": (row[1] or 0) > 0,
            "personal_score": row[2] or 0}


def _status_queue(dj: dict[str, Any], db_path=None) -> str | None:
    """The queue the status headline describes: the SAME ownership order
    the observer follows (ma_observer.resolve_target) -- the active
    listening session's queue first, then the selected output, then a
    single enabled output. `outputs.enabled[0]` is not where the DJ is
    working when the station plays on another enabled output, and a
    headline anchored there would report "idle" and a stale last break
    while breaks are airing. Read-only; any failure falls back to the old
    answer so status never raises."""
    outputs = dj.get("outputs") or {}
    try:
        import ma_observer
        c = _connect(db_path)
        try:
            sessions = [{"queue_id": r[0]} for r in c.execute(
                "SELECT queue_id FROM sessions WHERE status = 'active' "
                "AND queue_id IS NOT NULL AND queue_id != ''")]
        finally:
            c.close()
        target, _reason = ma_observer.resolve_target(sessions, outputs, None)
        if target:
            return target
    except Exception:
        logger.debug("dj status: ownership resolve failed", exc_info=True)
    enabled = outputs.get("enabled") or []
    return enabled[0] if enabled else None


def status_snapshot(now: datetime | None = None,
                    state_fn: Callable[[str], dict[str, Any]] | None = None,
                    db_path=None) -> dict[str, Any]:
    """The DJ's user-facing state, computed entirely from Music Brain truth.

    STRICTLY read-only: no settings writes, no break creation, no queue
    writes, no lifecycle mutation. The optional MA read (state_fn) is used
    only to estimate seconds-to-air from the current track's remaining time
    when the scheduler target alone cannot answer; any MA failure simply
    yields a null estimate.

    Statuses: off / idle / waiting / warming / preparing / ready / up_next /
    on_air, plus speech_off (DJ on, DJ speech switched off: no ordinary
    break is coming, so no target or ETA is published). ETA contract: estimated_seconds_to_air is a natural-boundary
    ESTIMATE -- present only when meaningful, 0 while on air, null when
    unknown."""
    now_dt = now or service_clock.legacy_now()
    try:
        import settings
        dj = settings.load()["dj"]
    except Exception:
        dj = {"enabled": False, "outputs": {"enabled": []}}
    enabled = bool(dj.get("enabled"))
    # Read from the same document as `enabled`, with the same rule as
    # settings.dj_speech_enabled: only an explicit false is a no.
    speech = dj.get("speech_enabled") is not False
    queue_id = _status_queue(dj, db_path)
    out: dict[str, Any] = {"enabled": enabled, "speech_enabled": speech,
                           "queue_id": queue_id,
                           "status": "off", "target_air_at": None,
                           "estimated_seconds_to_air": None,
                           "current_break": None, "last_break": None,
                           "skip_pending": None,
                           "current_track_reaction": None}
    if not enabled:
        return out

    c = _connect(db_path)
    try:
        session = None
        if queue_id:
            row = c.execute(
                "SELECT session_id, started_at, current_path, "
                "current_media_key FROM sessions WHERE status = 'active' "
                "AND queue_id = ? ORDER BY started_at DESC LIMIT 1",
                (queue_id,)).fetchone()
            if row:
                session = {"session_id": row[0], "started_at": row[1],
                           "current_path": row[2], "current_media_key": row[3]}
        brk_row = None
        if queue_id:
            r = c.execute(
                "SELECT token, state, telemetry, spoken_text, updated_at "
                "FROM dj_breaks WHERE queue_id = ? AND state IN "
                f"({','.join('?' for _ in NON_TERMINAL_STATES)}) "
                "ORDER BY created_at DESC LIMIT 1",
                (queue_id, *NON_TERMINAL_STATES)).fetchone()
            if r:
                try:
                    tel = json.loads(r[2] or "{}")
                except (TypeError, ValueError):
                    tel = {}
                brk_row = {"token": r[0], "state": r[1], "tel": tel,
                           "spoken_text": r[3] or "", "updated_at": r[4]}
        last_row = None
        if queue_id:
            r = c.execute(
                "SELECT spoken_text, updated_at, telemetry, token FROM dj_breaks "
                "WHERE queue_id = ? AND state = 'completed' "
                "ORDER BY updated_at DESC LIMIT 1", (queue_id,)).fetchone()
            if r:
                try:
                    tel = json.loads(r[2] or "{}")
                except (TypeError, ValueError):
                    tel = {}
                last_row = {"completed_at": r[1], "id": _public_ref(r[3]),
                            **_break_detail(tel, r[0] or "")}
                if tel.get("coverage"):   # not all of it was heard
                    last_row["coverage"] = tel["coverage"]["verdict"]
    finally:
        c.close()

    out["last_break"] = last_row
    if session is None:
        out["status"] = "idle"
        return out

    # Status from the live lifecycle row (warming is a planned intent whose
    # LLM warm-up already fired).
    if brk_row is None:
        out["status"] = "waiting"
    else:
        state = brk_row["state"]
        label = _STATUS_LABELS.get(state, "waiting")
        if state == "planned" and "llm_warm" in brk_row["tel"]:
            label = "warming"
        out["status"] = label
        out["current_break"] = {
            "token": brk_row["token"], "state": state,
            # The same public ref the program history gives this break once
            # it has aired, so a screen can tell the two are one break.
            "id": _public_ref(brk_row["token"]),
            # Whether these words are being heard RIGHT NOW, or are still
            # being prepared. A consumer must be able to tell a script that
            # is on air from one that is merely written.
            "on_air": label == "on_air",
            # Honest distinction for any consumer that cares: a liner is
            # not "the next scheduled DJ break".
            **_break_detail(brk_row["tel"], brk_row["spoken_text"])}

    # Scheduler-truth target: the intent's own target_air_at, else the same
    # anchor+cadence arithmetic plan_break will use for the next intent.
    target_iso = (brk_row or {}).get("tel", {}).get("target_air_at")
    if target_iso is None:
        anchor = max(filter(None, [session.get("started_at"),
                                   last_air_anchor(queue_id, db_path)]),
                     default=None)
        if anchor:
            try:
                target_iso = _iso(datetime.fromisoformat(anchor)
                                  + timedelta(minutes=cadence_minutes()))
            except (TypeError, ValueError):
                target_iso = None
    # A pending Cancel-Next-Break marker is user-visible truth: the next
    # ordinary opportunity is skipped, so the honest ETA is one cadence
    # after the suppressed slot.
    marker = get_skip_marker(queue_id, db_path) if queue_id else None
    if marker:
        out["skip_pending"] = {
            "suppressed_target": marker.get("suppressed_target")}
        suppressed = marker.get("suppressed_target")
        if brk_row is None and suppressed and (target_iso is None
                                               or target_iso <= suppressed):
            try:
                target_iso = _iso(service_clock.legacy_shift(
                    datetime.fromisoformat(suppressed),
                    timedelta(minutes=cadence_minutes())))
            except (TypeError, ValueError):
                pass
    overdue = (target_iso is not None and target_iso
               < service_clock.cutoff_destructive(now_dt, timedelta(0)))

    # ONE source of scheduling truth for what STATUS publishes. These two
    # keys describe the same event and used to be computed separately: the
    # slot's own `target_air_at` on one side, a natural-boundary estimate on
    # the other. When an `up_next` break's slot had already gone by, the
    # first was kept (the overdue guard excluded up_next/on_air) while the
    # second fell through to the boundary estimate -- so the panel could show
    # `next_break_at` in the PAST beside a positive `seconds_to_air`.
    #
    # The seconds are the answer. The timestamp is now + the seconds, so the
    # two cannot disagree, and both are null together when nothing credible
    # is known. A stale historical time is never published as a future one.
    if out["status"] == "on_air":
        eta = 0.0
    else:
        eta = None
        if target_iso and not overdue:
            try:
                eta = (datetime.fromisoformat(target_iso)
                       - now_dt).total_seconds()
            except (TypeError, ValueError):
                eta = None
        if eta is None or eta <= 0:
            # No usable slot target, or the slot already went by (a missed
            # break, DJ off through a slot, a queue rewind that moved the
            # boundary). The break airs at the next natural boundary, so the
            # current track's remaining time is the honest estimate.
            eta = None
            try:
                import ma_queue
                read_state = state_fn or ma_queue.queue_state
                eta = _remaining_s(read_state(queue_id))
            except Exception:
                eta = None
    out["estimated_seconds_to_air"] = int(eta) if eta is not None else None
    # Derived from the unrounded estimate: truncating to whole seconds first
    # would drift the timestamp by up to a second for no reason.
    out["target_air_at"] = (_iso(service_clock.legacy_shift(
        now_dt, timedelta(seconds=eta))) if eta is not None else None)

    # DJ SPEECH OFF: no ordinary break is coming, so none is promised.
    # Everything that WOULD have been reported about the next ordinary
    # break -- waiting on a slot, warming, a planned or rendered row the
    # next pass will close -- reads as `speech_off` with no target and no
    # ETA, and a skip marker the speech switch wrote is not presented as a
    # listener's Cancel. What is genuinely audible or already in Music
    # Assistant (on_air / up_next), and any non-ordinary row (a liner, a
    # news intro, an announcement), is still reported as it is.
    if not speech and out["status"] in ("waiting", "warming", "preparing",
                                         "ready"):
        if brk_row is None or is_ordinary_break(brk_row["tel"]):
            out["status"] = "speech_off"
            out["current_break"] = None
            out["estimated_seconds_to_air"] = None
            out["target_air_at"] = None
            if (marker or {}).get("source") == SPEECH_DISABLED:
                out["skip_pending"] = None

    # Current-track reaction: only for a RESOLVED ordinary music item; a
    # DJ break or unresolved item reads as null, never a guess.
    if out["status"] != "on_air":
        reaction = _reaction_for_path(session.get("current_path"))
        if reaction is not None:
            media_key = session.get("current_media_key") or None
            if media_key and media_key.startswith("file:"):
                media_key = None    # never expose filesystem-derived keys
            reaction["media_key"] = media_key
            out["current_track_reaction"] = reaction
    return out


# --- the ingestion hooks (called from playback.handle_event; never raise) -----

# Test seam: when set, ingestion-triggered hooks use these deps (tests set
# {"synchronous": True, "state_fn": <fake>} so fixture events can never spawn
# background scheduling threads or reach a real MA). Always None in
# production.
DEFAULT_DEPS: dict[str, Any] | None = None

def _news_pass(session: dict[str, Any], queue_id: str, now_dt: datetime,
               deps: dict[str, Any] | None, db_path) -> None:
    """Automatic news PLANNING rides the existing scheduler hooks -- no
    second timer, no second thread. Deliberately placed BEFORE the
    dj_enabled gate: news is its own feature and must not be silently
    coupled to whether the DJ is talking. It is inert while
    bulletins.enabled is false, returning before any network call or state
    write, and it is fenced so a planning fault can never disturb DJ
    scheduling or event ingestion. It never queues, revokes or arbitrates
    anything -- boundary ownership belongs to the on-air commit."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            import news_schedule
            news_schedule.scheduler_pass(session, queue_id, now_dt, deps,
                                         db_path=db_path)
        except Exception:
            logger.warning("news planning hook failed (music unaffected)",
                           exc_info=True)

    if deps.get("synchronous"):
        _run()                      # tests drive passes deliberately
        return
    # SINGLE FLIGHT. Off the ingestion thread, because a pass may render TTS
    # and that takes real seconds -- but at most ONE worker at a time. A
    # position tick arrives every few seconds, so spawning a thread per hook
    # would pile up dozens of workers all blocked on the planner's lock
    # behind one render, every one of them destined to re-do a pass whose
    # answer has not changed. Losing a tick costs nothing: the next one
    # re-evaluates from the same persisted state.
    if not _news_worker.acquire(blocking=False):
        return
    try:
        threading.Thread(target=_run_and_release, args=(_run,),
                         daemon=True).start()
    except Exception:               # thread could not start: never leak it
        _news_worker.release()
        raise


#: The most words a riding announcement may add to a break's budget.
ANNOUNCEMENT_ROOM_MAX = 60


def _bind_announcement_carrier(token: str, queue_id: str,
                               context: dict[str, Any], now_dt: datetime,
                               deps: dict[str, Any], db_path) -> None:
    """Calendar announcements: let a due plain ANNOUNCE ride this
    scheduled break (announcements.bind_for_dj_break), recording the link
    on the row BEFORE the render claim so the occurrence reconciles from
    the row whatever happens next. Inert unless combine_with_dj; fenced."""
    try:
        import announcements
        carrier = announcements.bind_for_dj_break(token, queue_id, now_dt,
                                                  deps, db_path)
    except Exception:  # noqa: BLE001
        logger.warning("announcement carrier check failed (break "
                       "unaffected)", exc_info=True)
        return
    if not carrier:
        return
    context["announcement_required"] = {
        "message": carrier["copy"], "occurrence_id": carrier["occurrence_id"]}
    _note(token, {"announcement": {k: v for k, v in carrier.items()
                                   if k != "copy"}}, db_path)


def _announcements_pass(session: dict[str, Any], queue_id: str,
                        now_dt: datetime, deps: dict[str, Any] | None,
                        db_path) -> None:
    """Announcement planning rides the same scheduler hooks as news: no
    second timer, no second thread pool. Placed BEFORE the dj_enabled
    gate on purpose -- an announcement is station programming, not DJ
    chatter, and must air with the DJ switched off. Inert while
    station.announcements.enabled is false (and whenever the occurrence
    store is empty), and fenced so an announcement fault can never
    disturb DJ scheduling or event ingestion. Single flight, exactly
    like the news worker: a pass may render TTS and takes real seconds."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            import announcements
            announcements.scheduler_pass(session, queue_id, now_dt, deps,
                                         db_path=db_path)
        except Exception:
            logger.warning("announcement planning hook failed "
                           "(music unaffected)", exc_info=True)

    if deps.get("synchronous"):
        _run()
        return
    if not _announcements_worker.acquire(blocking=False):
        return
    try:
        threading.Thread(target=_run_and_release_announcements,
                         args=(_run,), daemon=True).start()
    except Exception:               # thread could not start: never leak it
        _announcements_worker.release()
        raise


def _run_and_release_announcements(run: Callable[[], None]) -> None:
    try:
        run()
    finally:
        _announcements_worker.release()


def _run_and_release(run: Callable[[], None]) -> None:
    try:
        run()
    finally:
        _news_worker.release()


def _station_refill_pass(queue_id: str, now_dt: datetime,
                         deps: dict[str, Any] | None, db_path,
                         trigger: str = "tick") -> None:
    """Automatic station replenishment, riding the music-boundary hook.

    Deliberately placed BEFORE the dj_enabled gate, for the same reason
    _news_pass is: Keep Radio Playing is about whether the MUSIC keeps going,
    which has nothing to do with whether the DJ is allowed to speak. A station
    must not fall silent because the DJ was switched off.

    A track boundary is the only moment a runway actually shrinks, so this is
    complete coverage with no timer, no poller and no background worker. The
    eligibility policy decides; the transaction re-validates everything for
    itself and remains the authority on whether anything is written.

    SINGLE FLIGHT, and off the ingestion thread. Selection plus URI resolution
    is one serial Music Assistant search per pick and takes real seconds --
    far too long to sit in a timing-sensitive track-start path. At most one
    worker runs; losing a boundary costs nothing, because the next one
    re-evaluates from the same persisted state. The durable pending-batch
    uniqueness remains the real single-flight authority; this lock only avoids
    piling up threads.

    A Station Programmer (Mode 2) station is kept by station_live instead:
    it answers first, and the legacy evaluation below runs only when no
    such station is on air. This hook is also how a Mode 2 station on a
    queue the observer is NOT following hears about time passing -- the
    observed queue's ticks are its clock between reaper passes.

    Fenced completely: no refill problem may disturb event ingestion."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            import station_live
            if station_live.on_refill_hook(queue_id, now=now_dt,
                                           db_path=db_path, trigger=trigger):
                return
            import station_eligibility
            import station_refill
            decision = station_eligibility.evaluate(now=now_dt,
                                                    db_path=db_path)
            if not decision.eligible:
                logger.debug("station refill skipped on %s: %s",
                             queue_id, decision.reason)
                return
            logger.info(
                "station refill eligible: station=%s mode=%s owned=%s "
                "playable=%s desired=%s budget=%s",
                (decision.station_id or "")[:8], decision.mode,
                decision.owned_future, decision.playable_future,
                decision.desired_append_count, decision.selection_budget)
            result = station_refill.replenish_automatic(
                desired_append_count=decision.desired_append_count,
                selection_budget=decision.selection_budget,
                now=now_dt, db_path=db_path)
            logger.info(
                "station refill result: station=%s mode=%s feel=%s pass=%s "
                "selected=%s resolved=%s intended=%s appended=%s batch=%s "
                "ok=%s %s",
                (decision.station_id or "")[:8], decision.mode,
                result.get("feel"), result.get("relaxation_pass"),
                result.get("selected_count"), result.get("resolved_count"),
                result.get("intended_count"), result.get("appended_count"),
                str(result.get("batch_id") or "")[:8], result.get("ok"),
                result.get("reason") or "")
            prog = result.get("programmer")
            if prog and prog.get("status") not in (None, "disabled"):
                logger.info(
                    "station programmer: status=%s model=%s latency=%sms "
                    "candidates=%s order=%s",
                    prog.get("status"), prog.get("model"),
                    prog.get("latency_ms"), prog.get("candidate_count"),
                    " ".join(prog.get("order_after") or []) or "-")
            ap = result.get("artist_policy")
            if ap and ap.get("enabled") is not None:
                logger.info(
                    "artist policy: enabled=%s freq=%s budget=%s "
                    "candidate_pairs=%s final_pairs=%s max_run=%s "
                    "repaired=%s impossible=%s", ap.get("enabled"),
                    ap.get("frequency"), ap.get("budget"),
                    ap.get("candidate_pairs"), ap.get("final_pairs"),
                    ap.get("max_run"), ap.get("repaired"),
                    ap.get("perfect_impossible"))
            if result.get("selected_tracks"):
                logger.info("station refill picks (feel=%s): %s",
                            result.get("feel"),
                            "; ".join(f"{t.get('artist')} - {t.get('title')}"
                                      for t in result["selected_tracks"]))
        except Exception:
            logger.warning("station refill pass failed (music unaffected)",
                           exc_info=True)

    if deps.get("synchronous"):
        _run()                      # tests drive passes deliberately
        return
    if not _station_worker.acquire(blocking=False):
        return
    try:
        threading.Thread(target=_run_and_release_station, args=(_run,),
                         daemon=True).start()
    except Exception:               # thread could not start: never leak it
        _station_worker.release()
        raise


def _run_and_release_station(run: Callable[[], None]) -> None:
    try:
        run()
    finally:
        _station_worker.release()


# A queued break younger than this at a music boundary is not suspicious:
# insertion happens in a background pass and may race the very boundary that
# triggered it. One full track always exceeds this.
QUEUED_GRACE_S = 60.0


def reconcile_missed_queued(queue_id: str, now: datetime | None = None,
                            deps: dict[str, Any] | None = None,
                            db_path=None) -> bool:
    """Positive-evidence missed detection. True when a break was
    newly marked 'missed'.

    A queued break is the NEXT item on its queue, so the next boundary
    should be the break airing -- observed as a programming start, not a
    music start. A REAL MUSIC track starting while a break is still
    'queued' from before this boundary therefore means the queue either
    detoured (a listener inserted something ahead of the break, which will
    still air) or moved past the break without airing it (skip, rewind,
    flow restart -- all real MA behaviours). One ordered future read
    decides which: the break's token still ahead of the current item means
    leave it alone; absent means it can never air, and waiting the
    revoke_stale TTL (~30 min, a half-hour stale state) serves
    nobody. An unreadable queue changes nothing -- a closed gate is never
    evidence.

    Marking 'missed' promptly is what re-opens plan_break's one-in-flight
    gate, so the ordinary cadence replans a make-good naturally; nothing
    here creates a break, so no duplicate is possible."""
    now_dt = now or service_clock.legacy_now()
    deps = deps or DEFAULT_DEPS or {}
    c = _connect(db_path)
    try:
        row = c.execute(
            "SELECT token, updated_at FROM dj_breaks WHERE queue_id = ? "
            "AND state = 'queued'", (queue_id,)).fetchone()
    finally:
        c.close()
    if row is None:
        return False
    token, queued_at = row
    bounds = service_clock.age_bounds(queued_at, now_dt)
    age = bounds[0] if bounds is not None else QUEUED_GRACE_S + 1
    if age < QUEUED_GRACE_S:
        return False
    import ma_queue
    read_state = deps.get("state_fn") or ma_queue.queue_state
    read_future = deps.get("future_fn") or ma_queue.read_future_items
    try:
        state = read_state(queue_id)
        current = (state or {}).get("current_item") or {}
        if media_ref.break_token_of(current.get("uri")) == token:
            return False        # it IS on air; the airing hook owns it
        read = read_future(queue_id, current.get("queue_item_id"),
                           current_index=state.get("current_index"),
                           total_items=state.get("items"))
    except Exception as exc:  # noqa: BLE001 - closed-gate boundary
        logger.info("queued-break reconciliation: queue %s unreadable (%s)",
                    queue_id, exc)
        return False
    if not read.get("complete"):
        return False            # a partial view proves nothing
    for item in read.get("future") or ():
        uri = ((item.get("media_item") or {}).get("uri")
               if isinstance(item.get("media_item"), dict)
               else None) or item.get("uri")
        if media_ref.break_token_of(uri) == token:
            return False        # still ahead (a detour, not a skip)
    ok = _transition(token, ("queued",), "missed", now_dt, db_path)
    if ok:
        _note(token, {"missed_reason": "queue_moved_past_item"}, db_path)
        logger.info("break %s missed: queue %s moved past it without "
                    "airing (marked at the boundary, not by TTL)",
                    token[:8], queue_id)
    return ok



def _song_semantics_pass(queue_id: str, now_dt: datetime,
                         deps: dict[str, Any] | None, db_path) -> None:
    """Cache the playing track's lyric semantics while it plays.
    Fenced completely and off the ingestion thread; the pass itself yields
    to render pressure (spoken renders and, transitively, everything above
    it), so it can never cost a deadline. A break never waits on this --
    it only ever reads the cache."""
    deps = deps or DEFAULT_DEPS or {}

    def _run():
        try:
            import song_context
            song_context.opportunistic_pass(queue_id, now_dt, deps, db_path)
        except Exception:
            logger.warning("song semantics pass failed (nothing affected)",
                           exc_info=True)

    if deps.get("synchronous"):
        if deps.get("semantics"):        # tests opt IN; fixtures stay dark
            _run()
        return
    threading.Thread(target=_run, daemon=True).start()


def on_music_track_start(session: dict[str, Any], queue_id: str,
                         now: datetime | None = None,
                         deps: dict[str, Any] | None = None,
                         current: dict[str, Any] | None = None,
                         db_path=None) -> None:
    """Every real music boundary: complete any airing break, then run one
    airtime-aware scheduling pass (at_boundary=True: this is also the
    degraded render fallback when position ticks are not flowing, and the
    natural insertion moment for a ready break -- the structural gate is
    open at track start). Wrapped by the caller in a failure boundary;
    additionally self-contained here so no DJ problem can ever disturb
    event ingestion."""
    now_dt = now or service_clock.legacy_now()
    try:
        complete_airing(queue_id, now_dt, db_path)
        try:
            reconcile_missed_queued(queue_id, now_dt, deps, db_path)
        except Exception:
            logger.warning("queued-break reconciliation failed "
                           "(music unaffected)", exc_info=True)
        # Real music started: this is where a finished bulletin is observed
        # to have left the air, and where the post-news quiet guard opens
        # and later releases. Deliberately NOT gated on bulletins.enabled --
        # an event that already aired must complete its lifecycle even if
        # News is switched off mid-event.
        try:
            import news_schedule
            news_schedule.on_music_track_start(session, queue_id, now_dt,
                                               db_path)
        except Exception:
            logger.warning("news music hook failed (music unaffected)",
                           exc_info=True)
        _news_pass(session, queue_id, now_dt, deps, db_path)
        _announcements_pass(session, queue_id, now_dt, deps, db_path)
        # Automatic station replenishment. BEFORE the dj_enabled gate on
        # purpose: keeping the music going is not a speech decision.
        if queue_id:
            _station_refill_pass(queue_id, now_dt, deps, db_path,
                                 trigger="boundary")
            _song_semantics_pass(queue_id, now_dt, deps, db_path)
            _story_prep_pass(queue_id, now_dt, deps, db_path)
        if not queue_id or not dj_enabled_for_queue(queue_id):
            return
        deps = deps or DEFAULT_DEPS or {}

        def _pass():
            try:
                schedule_pass(queue_id, session, now_dt, deps,
                              at_boundary=True, db_path=db_path)
            except Exception:
                logger.warning("dj boundary pass failed (music unaffected)",
                               exc_info=True)

        if deps.get("synchronous"):
            _pass()
        else:
            threading.Thread(target=_pass, daemon=True).start()
    except Exception:
        logger.warning("dj engine hook failed (music unaffected)",
                       exc_info=True)


def on_playback_tick(session: dict[str, Any], queue_id: str,
                     now: datetime | None = None,
                     deps: dict[str, Any] | None = None,
                     db_path=None) -> None:
    """Every position heartbeat while music plays: one airtime-aware
    scheduling pass (at_boundary=False). Ticks are what make late-bound
    rendering possible -- the render window opens ~RENDER_LEAD_S before the
    estimated target boundary, mid-track, where only a tick can see it.
    Fenced exactly like the boundary hook: no DJ problem may ever disturb
    event ingestion."""
    _note_tick()
    now_dt = now or service_clock.legacy_now()
    try:
        _news_pass(session, queue_id, now_dt, deps, db_path)
        _announcements_pass(session, queue_id, now_dt, deps, db_path)
        # Automatic station replenishment. BEFORE the dj_enabled gate on
        # purpose: keeping the music going is not a speech decision.
        if queue_id:
            _station_refill_pass(queue_id, now_dt, deps, db_path)
            _song_semantics_pass(queue_id, now_dt, deps, db_path)
        if not queue_id or not dj_enabled_for_queue(queue_id):
            return
        deps = deps or DEFAULT_DEPS or {}

        def _pass():
            try:
                schedule_pass(queue_id, session, now_dt, deps,
                              at_boundary=False, db_path=db_path)
            except Exception:
                logger.warning("dj tick pass failed (music unaffected)",
                               exc_info=True)

        if deps.get("synchronous"):
            _pass()
        else:
            threading.Thread(target=_pass, daemon=True).start()
    except Exception:
        logger.warning("dj tick hook failed (music unaffected)",
                       exc_info=True)
