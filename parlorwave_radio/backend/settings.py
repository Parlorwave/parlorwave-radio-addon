#!/usr/bin/env python3
"""Music Brain settings: the one persistent, versioned configuration document.

Sectioned from the start:

  dj       -- persona-level product configuration (who the DJ is, when and
              how it speaks, where it may speak).
  playback -- engine-level playback behaviour the customer experiences.
  bulletins -- news bulletin production.
  station  -- station programming feel (how an active station's roster is
              played out). Global today; designed so a station profile can
              carry its own value later.

The sectioning is the point, not the count. Engine-level concerns (playback
lifecycle, station programming) get their own top-level sections when they
acquire real customer-facing settings; until then no section is invented for
them. Isolation is enforced by `save_section`, which may only write its own
section, so adding one later changes nothing about the existing route.

One file rather than several because atomic write, deep-merge, validation and
versioning already exist here exactly once; more documents would duplicate all
of it and add cross-file consistency questions for no gain.

Why a file and not sessions.db: sessions.db is operational state, explicitly
disposable, and `playback.ensure_sessions_schema` may rebuild it. Customer
settings must survive that, so they live in their own JSON document under the
service data root.

Reliability mechanisms are NOT settings. The inactivity lease and the progress
watchdog that bound zombie sessions stay in playback.py as env-overridable
constants: they are safety machinery, not preferences, and exposing them would
invite a customer to disable the thing that stops a dead session from living
forever.

There is deliberately no maximum-session-age setting. A playback attachment
ends when playback actually ends or stops demonstrating progress -- never
because a clock ran out. Programming identity, host shifts and narrative
context are separate lifetimes that will change on their own schedules without
disturbing a healthy queue.

Identity contract (binding). `outputs.enabled` and `outputs.default_request_output`
hold **Music Assistant player_ids**, never Home Assistant entity_ids:

  - MA player_id is MA's own stable key and survives HA entity renames.
  - Groups are first-class players in MA and therefore addressable identically.
  - A PlayerQueue's queue_id IS the player_id of the output that owns it, so
    the runtime permission check is a single containment test against this list
    -- and a child player can never escalate a group, because a grouped child's
    queue_id resolves to the GROUP's player_id, which is only present here if
    the customer enabled the group itself.
  - The same key works when Music Brain eventually subscribes to MA directly.

Human-readable names are display metadata resolved at the UI layer, never
identity.

This module owns validation and persistence only. Nothing here reads playback
state, and nothing here changes DJ behaviour -- consumers land in later steps.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from urllib.parse import urlsplit

import dj_persona
import news_bulletins
import news_sources
import news_schedule
import paths
import playback_arcs
import station_feel
import station_intent

logger = logging.getLogger("musicbrain")

SETTINGS_PATH = Path(os.environ.get(
    "MUSICBRAIN_SETTINGS", str(paths.BASE_DIR / "musicbrain_settings.json")))

DEFAULT_STATION_NAME = "ParlorWave Radio"
MAX_STATION_NAME_LENGTH = 60

# This format has never shipped. It is version 1 -- the pre-deployment shape of
# the development branch is not a public version anyone can be upgrading from.
SCHEMA_VERSION = 1

SECTIONS = ("dj", "playback", "bulletins", "station", "services")

#: Fields that are CREDENTIALS. They are stored, never returned, never
#: logged, and never exported. get_section redacts them on the way out and
#: replaces each with a `<field>_set` boolean, which is the only thing a
#: screen ever needs to know: "is one saved".
SECRET_FIELDS = frozenset({
    ("services", "voice_engine", "api_key"),
    ("services", "writer", "api_key"),
})

BULLETIN_MODES = (news_bulletins.ORIGIN_PUBLISHER,
                  news_bulletins.ORIGIN_GENERATED)

BREAK_LENGTHS = ("short", "normal", "long")
VOICE_SOURCES = ("builtin", "cloned", "licensed", "custom")
#: What a voice may be used for (dj.voice_scopes). Absent = "dj".
VOICE_SCOPES = ("dj", "announcement", "both")
#: DJ Shifts: the three shift names, as dj_context.SHIFTS spells them.
SHIFT_NAMES = ("morning", "afternoon", "night")
#: Bounds on station.announcements.combine_wait_minutes: how long a plain
#: ANNOUNCE may wait for a DJ break to carry it before it airs standalone.
COMBINE_WAIT_MIN = 1
COMBINE_WAIT_MAX = 30

#: Bound on a station.announcements.default_voice_id value.
MAX_VOICE_ID_LEN = 64
#: A custom slot's recorded authorization (dj.voice_slots.<slot>
#: .authorization): what an EMPTY one looks like, and the values a
#: filled one may carry. voice_library owns the meaning.
VOICE_AUTHORIZATION_EMPTY = {"status": "", "version": "", "granted_at": "",
                             "source": ""}
VOICE_AUTHORIZATION_STATUSES = ("", "granted")
VOICE_SAMPLE_SOURCES = ("", "recording", "file")
PERSONA_PROFILES = dj_persona.profile_ids()
CHATTINESS_LEVELS = dj_persona.CHATTINESS_LEVELS
STATION_FEELS = station_feel.FEELS

# The selectable resume-grace values (minutes). null/None = Never: a paused
# listening attachment is not ended merely because time passed. A fixed menu
# rather than a free number: these are customer choices, and the UI story
# ("5 / 15 / 30 / 60 / Never") should match the validation exactly.
RESUME_GRACE_CHOICES = (5, 15, 30, 60)

DEFAULTS: dict[str, Any] = {
    "version": SCHEMA_VERSION,
    "dj": {
        # Global customer intent. Gates BREAK INSERTION only, never observation.
        "enabled": False,
        # DJ SPEECH: may ordinary DJ hosting be spoken? Independent of
        # `enabled` and narrower than it. Off keeps the DJ's whole lifecycle
        # running -- observation, planning, cadence, station programming,
        # requests -- and stops only the DJ's own voice: scheduled and
        # manual breaks and the request acknowledgement liner. Each due
        # opportunity is closed with the reason `dj_speech_disabled`
        # instead of being written, voiced and inserted. It is NOT a
        # switch over all text-to-speech: news bulletins and their intro,
        # and calendar announcements, are station programming with their
        # own switches and are untouched by it. True by default, so an
        # installation that has never seen this key behaves as before.
        "speech_enabled": True,
        # MUSIC-AWARE STORIES -- TWO switches, deliberately separate:
        #   stories_enabled  the USER's preference. Biography-based
        #                    storytelling is a normal product feature, ON by
        #                    default; false is the listener's "no stories".
        #   stories_rollout  a TEMPORARY qualification gate, OFF by
        #                    default, opened only deliberately for a
        #                    supervised trial. It is not a preference and is
        #                    removed once the path is qualified.
        # The story path is live only when both are on
        # (dj_stories_active). Only an explicit false disables the
        # preference; only an explicit true opens the gate.
        "stories_enabled": True,
        "stories_rollout": False,
        # ATMOSPHERIC HOSTING (atmosphere.py), OFF by default: grounded
        # weather, daylight and time-of-day colour in story breaks -- now and
        # then, never in place of an owed report. Its own switch.
        "atmosphere": False,
        "outputs": {
            # MA player_ids (players and/or groups). Empty = DJ nowhere.
            "enabled": [],
            # Whether playback started outside ParlorWave may be adopted as a
            # session. Deliberately independent of dj.enabled: Music Brain
            # keeps learning from what you play even when the DJ is off.
            "auto_adopt_playback": True,
            # Where an unqualified request ("play Cold Orchard") lands.
            "default_request_output": None,
        },
        "persona": {
            "name": "DJ",
            # Built-in personality profile (dj_persona.PROFILES): delivery
            # style only, never facts. A SPEECH setting -- it reaches the
            # next rendered break and never touches playback or the
            # station.
            "profile": dj_persona.DEFAULT_PROFILE,
            # DEPRECATED, COMPATIBILITY ONLY (persona/airtime
            # decoupling). Kept in the schema and still validated so every
            # settings document written under the old UI -- null, "low",
            # "normal", "high" -- loads unchanged and no migration is
            # needed. It no longer decides anything: normal production
            # always runs at dj_persona.PRODUCTION_CHATTINESS, personality
            # no longer supplies a default, and no user-facing control
            # writes this key any more. Do not read it to steer a render.
            "chattiness": None,
            # Object, not a string: cloned/licensed voices and style
            # references slot in without a schema break.
            # id is the LOGICAL Music Brain voice profile, never a backend
            # asset name. refs carries source-specific resolution (e.g.
            # refs.chatterbox_voice = the Chatterbox asset). A development
            # install binds its own enrolled voice here; the product default
            # names only the logical shipped profile.
            "voice": {"id": "default_voice_1", "source": "builtin", "refs": {}},
        },
        # Voice library: per-slot display names for the three custom
        # voice slots. Identity is the slot key; the name is presentation
        # only and never reaches file or asset names. "" = unnamed.
        # identity: the sha256 hex fingerprint of the sample the slot
        # currently holds (voice_library.sample_identity), "" = none. A slot id is
        # reusable; this is how a decision about the slot tells the voice
        # it was made about from a later replacement.
        # authorization: the owner's recorded acknowledgement that they may
        # create and use this voice, written
        # by the enrollment that stored the sample and cleared with it.
        # status "granted" or "" (none recorded -- empty, or enrolled
        # before this was asked); version names the acknowledgement text
        # (voice_library.CONSENT_VERSION); granted_at is UTC ISO-8601;
        # source is how the sample arrived ("recording" in the Booth, or a
        # "file"). Nothing about the speaker beyond the display name.
        "voice_slots": {
            "custom_1": {"name": "", "identity": "",
                         "authorization": dict(VOICE_AUTHORIZATION_EMPTY)},
            "custom_2": {"name": "", "identity": "",
                         "authorization": dict(VOICE_AUTHORIZATION_EMPTY)},
            "custom_3": {"name": "", "identity": "",
                         "authorization": dict(VOICE_AUTHORIZATION_EMPTY)},
        },
        # Per-voice notes the owner writes to remember a voice ("calm,
        # slower reader"). Keyed by logical voice id; presentation only, never
        # reaches a prompt, an asset name or a provider. "" = no note.
        "voice_notes": {},
        # The owner's own display name for a built-in voice, keyed by voice
        # id. Presentation only; "" or absent = the curated name.
        "voice_names": {},
        # Delivery for the built-in voices, keyed by voice id: "speed"
        # (0.7..1.3) and "pause" between sentences (0.5..2.5), 1.0 = the
        # engine's own. Never applies to a cloned voice.
        "voice_tuning": {},
        # What a voice is AUTHORIZED for, keyed by
        # voice id. "dj" (the meaning of an absent entry: every shelved
        # voice has always been selectable as the DJ voice), "announcement"
        # (a saved voice a calendar event may request for an announcement,
        # and never the DJ's), or "both". A calendar event can only ever
        # name a voice whose scope here says announcement or both.
        "voice_scopes": {},
        # DJ Shifts: the DJ voice follows the station's three air shifts
        # (dj_context.SHIFTS, cut from the daypart boundaries) when
        # `enabled`. Each assignment is a logical voice id, and for a
        # custom voice the recording it names (identity, as
        # voice_slots.<slot>.identity) -- "" = unassigned. `applied` is
        # the shift occurrence ("2026-01-15:night") the selection was last
        # settled for: by the scheduler at a boundary, or by the owner
        # choosing a DJ, which then holds until the next boundary. Off by
        # default; off, nothing here is read.
        "shifts": {
            "enabled": False,
            "morning": {"voice_id": "", "identity": ""},
            "afternoon": {"voice_id": "", "identity": ""},
            "night": {"voice_id": "", "identity": ""},
            "applied": "",
        },
        "programming": {
            "default_mode": "default_adaptive",
            "cadence_min_minutes": 12,
            "break_length": "normal",
            "topics_per_break": 2,
        },
        "content": {"weather": True, "news": True, "calendar": True},
        "news": {"enabled": False, "sources": [], "window_minutes": [27, 37]},
        # What a break may SAY it remembers. Bounded independently of session
        # lifetime: a 24h session must never narrate 24h of counts. Narrative
        # memory is read by break grounding and never feeds session lifecycle.
        "narrative": {
            "window_tracks": 25,
            "window_hours": 3,
            "segment_gap_minutes": 45,
        },
    },
    # Engine-level playback behaviour -- the first non-DJ customer section.
    # Named for what the customer experiences, never for the machinery: this
    # is "how long does a pause stay resumable", not a lease or watchdog.
    # Internal reliability bounds (session lease, progress watchdog, reaper
    # interval, pause zombie bound) remain env-only in playback.py.
    "playback": {
        # Minutes a paused listening attachment stays resumable as the SAME
        # session. One of RESUME_GRACE_CHOICES, or null = Never (a real pause
        # is never ended by time alone; explicit stop, queue replacement and
        # disappeared-player recovery still apply).
        "resume_grace_minutes": 30,
        # One-shot resume of an OWNED station queue that MA left
        # idle at a rewound position after a flow restart. The instant
        # kill switch for the one transport command Music Brain has.
        "auto_recover_idle": True,
        # Keep Radio Playing -- "Automatically add more songs when the station
        # queue gets low." The customer switch for AUTOMATIC replenishment
        # only. Turning it off blocks automatic eligibility and nothing else:
        # the durable station stays active, its ownership witness and rotation
        # history are untouched, already-queued music keeps playing, confirmed
        # refills stand, and the explicit POST /station/replenish still works,
        # because asking for more music by hand is not automatic behaviour.
        "keep_radio_playing": True,
        # Queue hygiene: may Parlorwave remove its OWN stale
        # program items (aired breaks, spent bulletins) from the future of
        # the queue it is on? PERMISSION only -- "the operator allows it".
        # It is not the safety latch: the engine keeps a separate persistent
        # latch in the sessions database (queue_hygiene.py) that it trips
        # itself when a pass cannot prove the customer's music survived
        # untouched, and that latch is never cleared by editing this. Both
        # must agree before anything is removed. ON by default: on the
        # automatic triggers (first stable attach, position regression)
        # a pass removes only completed program
        # items, with every protected item kept in order -- so a
        # fresh install permits hygiene. An explicit false in a saved
        # document is the operator's choice and is never rewritten.
        "queue_hygiene": True,
        # PROTOTYPE, off by default: hand ParlorWave's own DJ
        # breaks and bulletins to Music Assistant as SOUND EFFECTS instead of
        # tracks. MA 2.10 never records a sound effect as an Autoplay seed,
        # never analyses it into Sonic Similarity and never puts it in the
        # playlog, so program audio cannot steer or become a music
        # recommendation. MA does not normalise a sound
        # effect, so with this on the station levels each clip itself toward
        # MA's -14 LUFS before it is servable, and only a levelled clip goes
        # in as a sound effect (program_loudness). Off until testing has
        # qualified Flow playback, grouping, loudness and the end of a queue
        # with it (ma_queue.program_media_ref).
        "program_audio_as_sound_effect": False,
    },
    # Station programming feel: which RefillPolicy the next replenishment
    # uses (station_feel). Applies to FUTURE refills of whatever station is
    # active; it never rebuilds the queue, removes queued songs or changes
    # the station's frozen artist roster.
    "station": {
        # The station's on-air spoken broadcast identity (name).
        "identity": {
            "name": DEFAULT_STATION_NAME,
        },
        "feel": station_feel.DEFAULT_FEEL,
        # The optional LLM radio programmer: reorder-only, failure-
        # safe. `enabled` is the instant kill switch; the deterministic
        # order is always the fallback. Model/timeout stay backend-side.
        "programmer": {
            "enabled": True,
            "model": "",             # "" = the configured fallback (config.py)
            "timeout_s": 12.0,
        },
        # Back-to-back same-artist programming. Governs ONLY the
        # runway ParlorWave autonomously programs -- never manual/adopted/album
        # playback. enabled False = spread artists apart whenever possible.
        # frequency (low/medium/high) tunes how often a permitted double
        # appears when enabled. Default enabled True + medium stays closest
        # to the behaviour without this knob: the selector already SOFTLY spaces
        # artists (a spacing weight), so same-artist doubles were uncommon
        # but possible and there was no hard back-to-back rule. Medium is a
        # permissive pair budget that rarely acts on the already-spaced
        # order, preserving the current feel while adding the customer knob;
        # Off would newly FORCE zero doubles, a bigger behaviour change.
        "same_artist": {
            "enabled": True,
            "frequency": "medium",
        },
        # Calendar-driven station announcements: a
        # calendar event whose summary begins ANNOUNCE / ANNOUNCE EXACT is
        # finite station programming, aired once per occurrence at a safe
        # music boundary. enabled gates airing only -- observation and the
        # Station-tab list keep working so the switch never hides state.
        # smart_rewriting gates the ONE optional LLM rewrite of plain
        # ANNOUNCE payloads. EXPERIMENTAL and OFF by default: plain
        # ANNOUNCE airs the deterministic renderer and ANNOUNCE EXACT the
        # operator's words, both complete paths with no model at all.
        # late_air_minutes bounds how long past its scheduled start an
        # occurrence may still air before it expires: 40 is conservative --
        # wide enough to wait out a news bulletin and a long song, narrow
        # enough that copy can never surface hours stale.
        # default_voice_id names the shelved voice every announcement is
        # spoken with when its event requests none; "" = the station's
        # own DJ voice, as before. A per-event
        # request that is authorized takes precedence over it. It is a
        # PREFERENCE, not a grant: the voice must itself be scoped for
        # announcements (dj.voice_scopes) or the render refuses it.
        # combine_with_dj: a plain
        # ANNOUNCE rides the next ordinary DJ break -- the DJ's own voice,
        # one queue item -- instead of airing as its own item. It waits for
        # that break at most combine_wait_minutes past its start, then airs
        # standalone exactly as before. ANNOUNCE EXACT is never combined.
        # OFF by default: with it off every announcement path is unchanged.
        "announcements": {
            "enabled": True,
            "smart_rewriting": False,
            "late_air_minutes": 40,
            "default_voice_id": "",
            "combine_with_dj": False,
            "combine_wait_minutes": 8,
        },
        # Dial Stations (Station Programmer): the saved
        # station presets a Dial position or the Station tab selects by id.
        # CONFIGURATION ONLY -- nothing in this block plays a preset.
        # The block carries its own version and is owned by station_intent:
        # it is validated by dial_config()/save() when a write touches it,
        # never by validate(), so a hand-damaged preset list can neither
        # block nor be "repaired" by a save of any other setting.
        "dial": deepcopy(station_intent.DIAL_DEFAULTS),
        # Station Programmer (Mode 2) speakers the OPERATOR has explicitly
        # authorized although they are house speakers: the house's DJ output
        # and/or a static member of a Music Assistant speaker group. EMPTY
        # by default and never inferred -- a speaker is listed only by a
        # deliberate settings write. Listing lifts exactly two refusals for
        # that speaker (target_is_house_output, target_in_group) and none of
        # the others: unknown, unavailable, a group itself, grouped or
        # synced right now (station_live.check_target).
        "live": {
            "authorized_targets": [],
        },
    },
    # News bulletins -- the Newsroom's own section. enabled gates LIVE
    # station participation ONLY: the Test Feed and Fetch actions a person
    # presses are never gated by it. mode picks how bulletins are produced:
    # publisher_audio (RSS enclosure download) today; generated_briefing
    # (a grounded script -> TTS) is an accepted, stored future mode that no
    # action implements yet.
    "bulletins": {
        "enabled": False,
        "mode": "publisher_audio",
        # What the station CALLS this source, in the owner's words. It is
        # spoken by the intro and shown in the queue, so it is presentation,
        # never identity -- identity is the canonicalised feed URL. Empty
        # means "derive something honest from the feed host".
        "source_name": "",
        # THE SOURCE LIST -- the Newsroom's real model since multi-source.
        # Each entry is {id, name, feed_url, enabled, target_minute}; a
        # station may carry a bulletin at :00 from one publisher and
        # another at :30 from a second. news_sources owns the shape, the
        # validation and the migration; nothing reads this key directly.
        #
        # EMPTY IS THE FACTORY STATE and is never an error. No publisher
        # ships preconfigured.
        #
        # An empty list here with a legacy `publisher.custom_feed_url`
        # still set means "not migrated yet", and news_sources.migrate
        # produces one source from the legacy fields. That is why removing
        # the last source also blanks those fields: otherwise the next read
        # would migrate the deleted feed straight back.
        "sources": [],
        "schedule": {
            # LEGACY, kept for one release. The single feed's minute of the
            # hour; each source now carries its own. Read only by the
            # migration, and still written so a rollback to the previous
            # build finds its feed where it left it.
            "target_minute": 30,
        },
        "publisher": {
            # PUBLIC DEFAULT IS UNCONFIGURED, ON PURPOSE. The built-in
            # registry in news_bulletins still exists for compatibility
            # with installs that already selected one, and a stored value
            # survives this default untouched (stored settings merge OVER
            # defaults). But a fresh install must not arrive with somebody
            # else's newsroom preselected: the Newsroom asks for a feed,
            # and says so until it has one.
            "source": "custom",         # "custom" | a built-in registry id
            "custom_feed_url": "",
        },
    },
}


#: OPTIONAL SERVICES, and optional is the whole point.
#:
#: Parlorwave is complete with the deterministic writer and a Home
#: Assistant text-to-speech voice. Neither of these makes anything work
#: that did not work before; each makes one part of it nicer, and each
#: falls back to what the base product already does when it is absent,
#: disabled, unreachable, slow or wrong.
#:
#: They are DELIBERATELY SEPARATE. The writer produces the words and the
#: voice engine speaks them; they fail independently, are configured
#: independently and are reported independently. One being down must never
#: be reported as the other being down.
DEFAULTS["services"] = {
    # The voice engine (a Chatterbox-compatible speech service).
    "voice_engine": {
        "enabled": False,
        "base_url": "",
        # Sent as a bearer token when set. NEVER returned by the API.
        "api_key": "",
        # The engine's own voice/asset name, when it offers a choice.
        "voice": "",
    },
    # The external writing model.
    "writer": {
        "enabled": False,
        # Which protocol to speak. Both are already implemented in the
        # backend; this chooses between them rather than describing a
        # vendor.
        "provider": "ollama",
        "base_url": "",
        "model": "",
        # NEVER returned by the API.
        "api_key": "",
    },
}

WRITER_PROVIDERS = ("ollama", "openai_compatible")


class SettingsError(ValueError):
    """Invalid settings payload. Message is safe to return to the caller."""


# Paths whose keys are deliberately open-ended. Everything else is closed, so a
# typo is a 422 instead of an inert value sitting in the document forever
# looking like it works.
#
# persona.voice.refs is the one extensible map: future consent recordings,
# style references and licence metadata are per-voice and provider-shaped, and
# enumerating them now would mean a schema change per voice provider.
EXTENSIBLE_PATHS: frozenset[tuple[str, ...]] = frozenset({
    ("dj", "persona", "voice", "refs"),
    # keyed by voice id: the set of voices is the library's business
    ("dj", "voice_notes"),
    ("dj", "voice_names"),
    ("dj", "voice_tuning"),
    ("dj", "voice_scopes"),
})

# Voice values the API's generic section routes may send back but never
# change. Each is settled by a voice_library rule validate() cannot see: a
# consent record exists only because an owner acknowledged CONSENT_VERSION
# while enrolling; an identity is the fingerprint of the sample a slot really
# holds, which every earlier decision about that slot is checked against; a
# scope is granted only to a shelved voice and never leaves the DJ voice
# announcement-only; the DJ voice and the shift plan are picked among the
# voices allowed to be the DJ and bind the recording they name; the
# announcement default must already be allowed for announcements.
# voice_library writes them through save_section like any other caller --
# only save_section(generic=True) is held to this.
#
# Keyed by path, "*" standing for every key of that map; the value names the
# route that makes the change instead.
VOICE_OWNED_PATHS: dict[tuple[str, ...], str] = {
    ("dj", "voice_scopes"): "POST /voices/{voice_id}/scope",
    ("dj", "voice_slots", "*", "identity"):
        "the Voice Booth's recording routes (/voices/custom/...)",
    ("dj", "voice_slots", "*", "authorization"):
        "the Voice Booth's recording routes (/voices/custom/...)",
    ("dj", "persona", "voice", "id"): "POST /voices/select",
    ("dj", "persona", "voice", "source"): "POST /voices/select",
    ("dj", "persona", "voice", "refs", "*"): "POST /voices/select",
    ("dj", "shifts"): "POST /voices/shifts",
    ("station", "announcements", "default_voice_id"):
        "POST /voices/announcement-default",
}
# ...apart from the Studio's original voice write (the integration's
# set_voice): a Chatterbox asset name, which PUT /settings/dj checks against
# the live voice list.
VOICE_OWNED_EXEMPT: frozenset[tuple[str, ...]] = frozenset({
    ("dj", "persona", "voice", "refs", "chatterbox_voice"),
})


def _reject_unknown(patch: Any, allowed: Any, path: tuple[str, ...]) -> None:
    """Recursively reject keys the v1 schema does not define.

    Walks the PATCH, not the merged document: a stored file may still carry
    keys from a future version (load stays lenient so a newer release's
    document cannot brick an older service), but nothing unknown may ENTER
    through the API.
    """
    if not isinstance(patch, dict):
        return
    if not isinstance(allowed, dict):
        # A scalar/list leaf in the schema was sent an object. Leave the type
        # complaint to validate(), which words it better.
        return
    for key, value in patch.items():
        here = path + (key,)
        if key not in allowed:
            known = ", ".join(sorted(allowed)) or "(none)"
            raise SettingsError(
                f"unknown settings field: {'.'.join(here)} "
                f"(known fields here: {known})")
        if here in EXTENSIBLE_PATHS:
            continue
        _reject_unknown(value, allowed[key], here)


def _voice_owned(doc: dict[str, Any], section: str
                 ) -> dict[str, tuple[str, Any]]:
    """{dotted path: (owning route, value)} for the voice-owned values in
    one section of `doc`. A container that is not an object holds none
    (validate() names it), so a write that replaces one shows up as every
    value it would take away."""
    found: dict[str, tuple[str, Any]] = {}

    def walk(node: Any, rest: tuple[str, ...], here: tuple[str, ...],
             route: str) -> None:
        if here in VOICE_OWNED_EXEMPT:
            return
        if not rest:
            found[".".join(here)] = (route, node)
        elif isinstance(node, dict):
            keys = list(node) if rest[0] == "*" else \
                [rest[0]] if rest[0] in node else []
            for key in keys:
                walk(node[key], rest[1:], here + (key,), route)

    for owned, route in VOICE_OWNED_PATHS.items():
        if owned[0] == section:
            walk(doc.get(section), owned[1:], owned[:1], route)
    return found


def _refuse_voice_owned_change(section: str, patch: dict[str, Any],
                               path: Path | None = None) -> None:
    """A generic-route write may carry a voice-owned value only to send it
    back as it is. Judged on the merged RESULT rather than on the patch's
    spelling, so no nesting, partial record or replaced container gets past
    it -- and against the document as stored NOW, so a stale read sent back
    cannot quietly undo a voice decision made since it was read.

    A value the stored document does not hold at all (a hand edit left its
    container something other than an object) reads as its schema default:
    an ordinary save repairs such a document -- _migrate fills the defaults
    back in, and "save the Station settings again" is the announcements
    remedy for it -- and a default grants nothing. A stored value is never
    replaced by its default here: that would erase a consent record."""
    stored = load(path)
    before = _voice_owned(stored, section)
    default = _voice_owned(_migrate({}), section)
    after = _voice_owned(_migrate(_merge(stored, {section: patch})), section)
    for where in sorted(before.keys() | after.keys()):
        old, new = before.get(where) or default.get(where), after.get(where)
        if old is None or new is None or (
                json.dumps(old[1], sort_keys=True)
                != json.dumps(new[1], sort_keys=True)):
            raise SettingsError(f"{where} is saved through {(old or new)[0]}, "
                                f"not the {section} settings")


# --- validation ---------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SettingsError(msg)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def validate(doc: dict[str, Any]) -> None:
    """Validate a FULLY MERGED document. Raises SettingsError on the first
    problem, with a message naming the field."""
    _require(isinstance(doc, dict), "settings must be an object")

    dj = doc.get("dj")
    _require(isinstance(dj, dict), "dj must be an object")
    _require(isinstance(dj.get("enabled"), bool), "dj.enabled must be a boolean")
    _require(isinstance(dj.get("speech_enabled"), bool),
             "dj.speech_enabled must be a boolean")
    _require(isinstance(dj.get("stories_enabled"), bool),
             "dj.stories_enabled must be a boolean")
    _require(isinstance(dj.get("stories_rollout"), bool),
             "dj.stories_rollout must be a boolean")
    _require(isinstance(dj.get("atmosphere"), bool),
             "dj.atmosphere must be a boolean")

    out = dj.get("outputs")
    _require(isinstance(out, dict), "dj.outputs must be an object")
    _require(_is_str_list(out.get("enabled")),
             "dj.outputs.enabled must be a list of non-empty Music Assistant player_ids")
    _require(len(set(out["enabled"])) == len(out["enabled"]),
             "dj.outputs.enabled must not contain duplicates")
    _require(isinstance(out.get("auto_adopt_playback"), bool),
             "dj.outputs.auto_adopt_playback must be a boolean")
    dro = out.get("default_request_output")
    _require(dro is None or (isinstance(dro, str) and dro.strip()),
             "dj.outputs.default_request_output must be a player_id or null")
    # The invariant, asserted as well as resolved: hiding a speaker from a
    # dropdown is not the same as refusing to DJ on it, and this is the
    # half that holds when the write did not come from the panel.
    _require(dro is None or dro in (out.get("enabled") or []),
             "dj.outputs.default_request_output must be one of "
             "dj.outputs.enabled (the speakers ParlorWave may DJ on)")

    per = dj.get("persona")
    _require(isinstance(per, dict), "dj.persona must be an object")
    _require(isinstance(per.get("name"), str) and per["name"].strip(),
             "dj.persona.name must be a non-empty string")
    _require(per.get("profile") in PERSONA_PROFILES,
             "dj.persona.profile must be one of "
             f"{', '.join(PERSONA_PROFILES)}")
    # Deprecated compatibility key: still accepted so an old document is
    # still a legal one. Nothing in normal production reads it.
    chat = per.get("chattiness")
    _require(chat is None or chat in CHATTINESS_LEVELS,
             "dj.persona.chattiness is deprecated and no longer affects "
             "the DJ; when present it must still be one of "
             f"{', '.join(CHATTINESS_LEVELS)}, or null")
    voice = per.get("voice")
    _require(isinstance(voice, dict), "dj.persona.voice must be an object")
    _require(isinstance(voice.get("id"), str) and voice["id"].strip(),
             "dj.persona.voice.id must be a non-empty string")
    _require(voice.get("source") in VOICE_SOURCES,
             f"dj.persona.voice.source must be one of {', '.join(VOICE_SOURCES)}")
    _require(isinstance(voice.get("refs"), dict),
             "dj.persona.voice.refs must be an object")

    slots = dj.get("voice_slots")
    _require(isinstance(slots, dict), "dj.voice_slots must be an object")
    for key in ("custom_1", "custom_2", "custom_3"):
        slot = slots.get(key)
        _require(isinstance(slot, dict), f"dj.voice_slots.{key} must be an "
                 "object")
        name = slot.get("name")
        _require(isinstance(name, str) and len(name) <= 40,
                 f"dj.voice_slots.{key}.name must be a string of at most "
                 "40 characters")
        ident = slot.get("identity", "")
        _require(isinstance(ident, str) and len(ident) <= 64
                 and (not ident or ident.isalnum()),
                 f"dj.voice_slots.{key}.identity must be a short "
                 "alphanumeric fingerprint or empty")
        auth = slot.get("authorization", VOICE_AUTHORIZATION_EMPTY)
        _require(isinstance(auth, dict)
                 and set(auth) <= set(VOICE_AUTHORIZATION_EMPTY),
                 f"dj.voice_slots.{key}.authorization must be an object "
                 "with status, version, granted_at and source")
        _require(auth.get("status", "") in VOICE_AUTHORIZATION_STATUSES,
                 f"dj.voice_slots.{key}.authorization.status must be "
                 "'granted' or empty")
        _require(auth.get("source", "") in VOICE_SAMPLE_SOURCES,
                 f"dj.voice_slots.{key}.authorization.source must be "
                 "'recording', 'file' or empty")
        for field in ("version", "granted_at"):
            val = auth.get(field, "")
            _require(isinstance(val, str) and len(val) <= 40
                     and val.isprintable(),
                     f"dj.voice_slots.{key}.authorization.{field} must be "
                     "a short string")

    notes = dj.get("voice_notes")
    _require(isinstance(notes, dict), "dj.voice_notes must be an object")
    for key, note in notes.items():
        _require(isinstance(key, str) and 1 <= len(key) <= 32
                 and key.replace("_", "a").isalnum() and key == key.lower(),
                 "dj.voice_notes keys must be lowercase voice ids")
        _require(isinstance(note, str) and len(note) <= 80,
                 f"dj.voice_notes.{key} must be a string of at most 80 "
                 "characters")

    names = dj.get("voice_names")
    _require(isinstance(names, dict), "dj.voice_names must be an object")
    for key, name in names.items():
        _require(isinstance(key, str) and 1 <= len(key) <= 32
                 and key.replace("_", "a").isalnum() and key == key.lower(),
                 "dj.voice_names keys must be lowercase voice ids")
        _require(isinstance(name, str) and len(name) <= 40,
                 f"dj.voice_names.{key} must be a string of at most 40 "
                 "characters")

    tuning = dj.get("voice_tuning")
    _require(isinstance(tuning, dict), "dj.voice_tuning must be an object")
    for key, tune in tuning.items():
        _require(isinstance(key, str) and 1 <= len(key) <= 32
                 and key.replace("_", "a").isalnum() and key == key.lower(),
                 "dj.voice_tuning keys must be lowercase voice ids")
        _require(isinstance(tune, dict) and set(tune) <= {"speed", "pause"},
                 f"dj.voice_tuning.{key} may only carry speed and pause")
        for field, (lo, hi) in (("speed", (0.7, 1.3)), ("pause", (0.5, 2.5))):
            if field in tune:
                val = tune[field]
                _require(isinstance(val, (int, float)) and not isinstance(val, bool)
                         and lo <= val <= hi,
                         f"dj.voice_tuning.{key}.{field} must be a number between "
                         f"{lo} and {hi}")

    shifts = dj.get("shifts")
    _require(isinstance(shifts, dict), "dj.shifts must be an object")
    _require(isinstance(shifts.get("enabled"), bool),
             "dj.shifts.enabled must be a boolean")
    for name in SHIFT_NAMES:
        slot = shifts.get(name)
        _require(isinstance(slot, dict) and set(slot) <= {"voice_id", "identity"},
                 f"dj.shifts.{name} must be an object with voice_id and identity")
        vid = slot.get("voice_id", "")
        _require(isinstance(vid, str) and len(vid) <= MAX_VOICE_ID_LEN
                 and (not vid or (vid.replace("_", "a").isalnum()
                                  and vid == vid.lower())),
                 f"dj.shifts.{name}.voice_id must be a lowercase voice id or empty")
        ident = slot.get("identity", "")
        _require(isinstance(ident, str) and len(ident) <= 64
                 and (not ident or ident.isalnum()),
                 f"dj.shifts.{name}.identity must be a short alphanumeric "
                 "fingerprint or empty")
    applied = shifts.get("applied")
    _require(isinstance(applied, str) and len(applied) <= 24
             and (not applied or re.fullmatch(
                 r"\d{4}-\d{2}-\d{2}:(" + "|".join(SHIFT_NAMES) + ")", applied)),
             "dj.shifts.applied must be a shift occurrence like "
             "2026-09-30:night, or empty")

    scopes = dj.get("voice_scopes")
    _require(isinstance(scopes, dict), "dj.voice_scopes must be an object")
    for key, scope in scopes.items():
        _require(isinstance(key, str) and 1 <= len(key) <= 32
                 and key.replace("_", "a").isalnum() and key == key.lower(),
                 "dj.voice_scopes keys must be lowercase voice ids")
        _require(scope in VOICE_SCOPES,
                 f"dj.voice_scopes.{key} must be one of "
                 f"{', '.join(VOICE_SCOPES)}")

    prog = dj.get("programming")
    _require(isinstance(prog, dict), "dj.programming must be an object")
    _require(prog.get("default_mode") in playback_arcs.VALID_MODES,
             "dj.programming.default_mode must be a known playback mode "
             f"({', '.join(sorted(playback_arcs.VALID_MODES))})")
    cad = prog.get("cadence_min_minutes")
    # ONE minute is the floor, and the message says so. The bound used to
    # be `0 < cad` under a message that said "between 0 and 240", so the
    # service refused 0 and 0.5 while every surface advertised a minimum of
    # zero and the error copy agreed with none of it. Zero is deliberately
    # not a value: "the DJ does not speak" is dj.speech_enabled (or
    # dj.enabled, which stops the whole DJ), and a cadence of
    # nothing would be a second, quieter way to say it.
    _require(isinstance(cad, (int, float)) and not isinstance(cad, bool)
             and 1 <= cad <= 240,
             "dj.programming.cadence_min_minutes must be a number between 1 and 240")
    _require(prog.get("break_length") in BREAK_LENGTHS,
             f"dj.programming.break_length must be one of {', '.join(BREAK_LENGTHS)}")
    tpb = prog.get("topics_per_break")
    _require(isinstance(tpb, int) and not isinstance(tpb, bool) and 1 <= tpb <= 4,
             "dj.programming.topics_per_break must be an integer between 1 and 4")

    con = dj.get("content")
    _require(isinstance(con, dict), "dj.content must be an object")
    for key in ("weather", "news", "calendar"):
        _require(isinstance(con.get(key), bool), f"dj.content.{key} must be a boolean")

    news = dj.get("news")
    _require(isinstance(news, dict), "dj.news must be an object")
    _require(isinstance(news.get("enabled"), bool), "dj.news.enabled must be a boolean")
    _require(_is_str_list(news.get("sources")) or news.get("sources") == [],
             "dj.news.sources must be a list of non-empty strings")
    win = news.get("window_minutes")
    _require(isinstance(win, list) and len(win) == 2
             and all(isinstance(x, int) and not isinstance(x, bool) for x in win),
             "dj.news.window_minutes must be [start, end] integers")
    _require(0 <= win[0] <= win[1] <= 59,
             "dj.news.window_minutes must satisfy 0 <= start <= end <= 59")

    nar = dj.get("narrative")
    _require(isinstance(nar, dict), "dj.narrative must be an object")
    wt = nar.get("window_tracks")
    _require(isinstance(wt, int) and not isinstance(wt, bool) and 1 <= wt <= 500,
             "dj.narrative.window_tracks must be an integer between 1 and 500")
    wh = nar.get("window_hours")
    _require(isinstance(wh, (int, float)) and not isinstance(wh, bool) and 0 < wh <= 48,
             "dj.narrative.window_hours must be a number between 0 and 48")
    gap = nar.get("segment_gap_minutes")
    _require(isinstance(gap, int) and not isinstance(gap, bool) and 1 <= gap <= 720,
             "dj.narrative.segment_gap_minutes must be an integer between 1 and 720")

    pb = doc.get("playback")
    _require(isinstance(pb, dict), "playback must be an object")
    grace = pb.get("resume_grace_minutes")
    _require(grace is None or (isinstance(grace, int) and not isinstance(grace, bool)
                               and grace in RESUME_GRACE_CHOICES),
             "playback.resume_grace_minutes must be one of "
             f"{', '.join(str(v) for v in RESUME_GRACE_CHOICES)}, or null (never)")
    _require(isinstance(pb.get("keep_radio_playing"), bool),
             "playback.keep_radio_playing must be a boolean")
    _require(isinstance(pb.get("auto_recover_idle"), bool),
             "playback.auto_recover_idle must be a boolean")
    _require(isinstance(pb.get("queue_hygiene"), bool),
             "playback.queue_hygiene must be a boolean")
    _require(isinstance(pb.get("program_audio_as_sound_effect"), bool),
             "playback.program_audio_as_sound_effect must be a boolean")

    st = doc.get("station")
    _require(isinstance(st, dict), "station must be an object")
    ident = st.get("identity")
    _require(isinstance(ident, dict), "station.identity must be an object")
    s_name = ident.get("name")
    _require(isinstance(s_name, str), "station.identity.name must be a string")
    _require(len(s_name.strip()) <= MAX_STATION_NAME_LENGTH,
             f"station.identity.name must be at most {MAX_STATION_NAME_LENGTH} characters")
    _require("\n" not in s_name and "\r" not in s_name,
             "station.identity.name must be a single line")
    _require(not ("<" in s_name or ">" in s_name),
             "station.identity.name must not contain markup")
    _require(all(ord(c) >= 32 and not unicodedata.category(c).startswith("C") for c in s_name),
             "station.identity.name must not contain control characters")
    _require(st.get("feel") in STATION_FEELS,
             f"station.feel must be one of {', '.join(STATION_FEELS)}")
    prog = st.get("programmer")
    _require(isinstance(prog, dict), "station.programmer must be an object")
    _require(isinstance(prog.get("enabled"), bool),
             "station.programmer.enabled must be a boolean")
    _require(isinstance(prog.get("model"), str),
             "station.programmer.model must be a string (empty = the "
             "configured default)")
    timeout = prog.get("timeout_s")
    _require(isinstance(timeout, (int, float))
             and not isinstance(timeout, bool) and 1.0 <= timeout <= 60.0,
             "station.programmer.timeout_s must be a number in [1, 60]")
    live = st.get("live")
    _require(isinstance(live, dict), "station.live must be an object")
    auth = live.get("authorized_targets")
    _require(_is_str_list(auth) and len(set(auth)) == len(auth)
             and len(auth) <= 8,
             "station.live.authorized_targets must be a list of distinct "
             "Music Assistant player_ids (at most 8)")
    sa = st.get("same_artist")
    _require(isinstance(sa, dict), "station.same_artist must be an object")
    _require(isinstance(sa.get("enabled"), bool),
             "station.same_artist.enabled must be a boolean")
    _require(sa.get("frequency") in ("low", "medium", "high"),
             "station.same_artist.frequency must be low, medium or high")
    ann = st.get("announcements")
    _require(isinstance(ann, dict), "station.announcements must be an object")
    _require(isinstance(ann.get("enabled"), bool),
             "station.announcements.enabled must be a boolean")
    _require(isinstance(ann.get("smart_rewriting"), bool),
             "station.announcements.smart_rewriting must be a boolean")
    late = ann.get("late_air_minutes")
    _require(isinstance(late, int) and not isinstance(late, bool)
             and 5 <= late <= 120,
             "station.announcements.late_air_minutes must be an integer "
             "in [5, 120]")
    dvoice = ann.get("default_voice_id")
    _require(isinstance(dvoice, str) and len(dvoice) <= MAX_VOICE_ID_LEN,
             "station.announcements.default_voice_id must be a string of at "
             f"most {MAX_VOICE_ID_LEN} characters (empty for the DJ voice)")
    _require(isinstance(ann.get("combine_with_dj"), bool),
             "station.announcements.combine_with_dj must be a boolean")
    cwait = ann.get("combine_wait_minutes")
    _require(isinstance(cwait, int) and not isinstance(cwait, bool)
             and COMBINE_WAIT_MIN <= cwait <= COMBINE_WAIT_MAX,
             "station.announcements.combine_wait_minutes must be an integer "
             f"in [{COMBINE_WAIT_MIN}, {COMBINE_WAIT_MAX}]")

    sv = doc.get("services")
    _require(isinstance(sv, dict), "services must be an object")
    ve = sv.get("voice_engine")
    _require(isinstance(ve, dict), "services.voice_engine must be an object")
    _require(isinstance(ve.get("enabled"), bool),
             "services.voice_engine.enabled must be a boolean")
    _require_service_url(ve.get("base_url"), "services.voice_engine.base_url")
    _require(isinstance(ve.get("api_key"), str) and len(ve["api_key"]) <= 500,
             "services.voice_engine.api_key must be a string of at most 500 "
             "characters")
    _require(isinstance(ve.get("voice"), str) and len(ve["voice"]) <= 120,
             "services.voice_engine.voice must be a string of at most 120 "
             "characters")
    # An enabled service with nowhere to reach is a promise the station
    # cannot keep, and it is better refused while somebody is looking at
    # the box than discovered at the next break.
    _require(not ve["enabled"] or ve.get("base_url", "").strip(),
             "services.voice_engine needs an address before it can be "
             "switched on")

    wr = sv.get("writer")
    _require(isinstance(wr, dict), "services.writer must be an object")
    _require(isinstance(wr.get("enabled"), bool),
             "services.writer.enabled must be a boolean")
    _require(wr.get("provider") in WRITER_PROVIDERS,
             "services.writer.provider must be one of "
             + ", ".join(WRITER_PROVIDERS))
    _require_service_url(wr.get("base_url"), "services.writer.base_url")
    _require(isinstance(wr.get("model"), str) and len(wr["model"]) <= 120,
             "services.writer.model must be a string of at most 120 characters")
    _require(isinstance(wr.get("api_key"), str) and len(wr["api_key"]) <= 500,
             "services.writer.api_key must be a string of at most 500 characters")
    _require(not wr["enabled"] or wr.get("base_url", "").strip(),
             "services.writer needs an address before it can be switched on")
    _require(not wr["enabled"] or wr.get("model", "").strip(),
             "services.writer needs a model name before it can be switched on")

    bl = doc.get("bulletins")
    _require(isinstance(bl, dict), "bulletins must be an object")
    _require(isinstance(bl.get("enabled"), bool),
             "bulletins.enabled must be a boolean")
    _require(bl.get("mode") in BULLETIN_MODES,
             f"bulletins.mode must be one of {', '.join(BULLETIN_MODES)}")
    name = bl.get("source_name")
    _require(isinstance(name, str) and len(name) <= 60,
             "bulletins.source_name must be a string of at most 60 characters")
    _require("\n" not in name and "\r" not in name,
             "bulletins.source_name must be a single line")
    try:
        news_sources.validate_sources(bl.get("sources", []))
    except news_sources.SourceError as exc:
        raise SettingsError(str(exc))
    sched = bl.get("schedule")
    _require(isinstance(sched, dict), "bulletins.schedule must be an object")
    tm = sched.get("target_minute")
    _require(isinstance(tm, int) and not isinstance(tm, bool)
             and tm in news_schedule.SUPPORTED_TARGET_MINUTES,
             "bulletins.schedule.target_minute must be one of "
             + ", ".join(str(m) for m in news_schedule.SUPPORTED_TARGET_MINUTES))
    pub = bl.get("publisher")
    _require(isinstance(pub, dict), "bulletins.publisher must be an object")
    src = pub.get("source")
    # The public build ships no built-in publisher, so 'custom' is the only
    # source a public installation can write. The registry lookup stays: a
    # future Parlorwave-produced preset becomes valid here by existing.
    _builtin = sorted(news_bulletins.BUILTIN_SOURCES)
    _require(src == "custom" or src in news_bulletins.BUILTIN_SOURCES,
             "bulletins.publisher.source must be 'custom'"
             + (f" or a built-in source id ({', '.join(_builtin)})"
                if _builtin else ""))
    url = pub.get("custom_feed_url")
    _require(isinstance(url, str) and len(url) <= 500,
             "bulletins.publisher.custom_feed_url must be a string of at "
             "most 500 characters")
    if url.strip():
        parts = urlsplit(url.strip())
        _require(parts.scheme in ("http", "https") and bool(parts.netloc),
                 "bulletins.publisher.custom_feed_url must be an http(s) URL")
        # A URL with no host, or one carrying credentials, is a typo or a
        # mistake worth refusing while a person is still looking at the box.
        _require(bool(parts.hostname),
                 "bulletins.publisher.custom_feed_url must name a host")
        _require("@" not in parts.netloc,
                 "bulletins.publisher.custom_feed_url must not carry "
                 "credentials in the URL")


# --- persistence --------------------------------------------------------------

def _require_service_url(url: Any, where: str) -> None:
    """An optional service address. Empty is always fine -- that is 'not
    configured', which is the normal state."""
    _require(isinstance(url, str) and len(url) <= 500,
             f"{where} must be a string of at most 500 characters")
    if not url.strip():
        return
    parts = urlsplit(url.strip())
    _require(parts.scheme in ("http", "https") and bool(parts.netloc),
             f"{where} must be an http(s) URL")
    _require(bool(parts.hostname), f"{where} must name a host")
    # Credentials in a URL end up in error messages, in logs and in
    # anything that echoes the address back. The key field is the place
    # for a key.
    _require("@" not in parts.netloc,
             f"{where} must not carry credentials in the URL -- use the "
             f"key field, which is never shown or logged")


def redacted(doc: dict[str, Any]) -> dict[str, Any]:
    """A copy with every credential removed and replaced by a boolean.

    THE ONLY SHAPE THAT LEAVES THIS MODULE towards an API, a diagnostic or
    an export. A screen never needs a key back -- it needs to know whether
    one is saved -- so `api_key` becomes None and `api_key_set` says the
    rest. Returning a key "so the form can show it" is how a credential
    ends up in a browser, in a screenshot and in a bug report.
    """
    out = deepcopy(doc)
    for path in SECRET_FIELDS:
        node = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                break
        else:
            field = path[-1]
            if field in node:
                node[f"{field}_set"] = bool(str(node.get(field) or "").strip())
                node[field] = None
    return out


def _keep_unchanged_secrets(section: str, patch: dict[str, Any],
                            stored: dict[str, Any]) -> None:
    """A patch that does not mention a secret must not erase it.

    The API returns None for every key, so a screen that read the section
    and sent it back would otherwise blank the credential it never saw.
    null means LEAVE IT; an empty string is the deliberate way to clear
    one, and is the only way.
    """
    for path in SECRET_FIELDS:
        if path[0] != section:
            continue
        # Both walks start INSIDE the section, so they stay in step.
        node = patch
        kept = stored.get(section) or {}
        for key in path[1:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            kept = kept.get(key) if isinstance(kept, dict) else {}
        if not isinstance(node, dict):
            continue
        field = path[-1]
        if node.get(field, "unset") is None:
            node[field] = str((kept or {}).get(field) or "")
        # <field>_set is a READ-ONLY reflection this module adds on the way
        # out; a screen that sends the section back must not be refused for
        # echoing it.
        node.pop(f"{field}_set", None)


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge. Lists REPLACE wholesale -- a caller sending
    outputs.enabled means exactly that list, never an append."""
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else SETTINGS_PATH


def _migrate(doc: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored document up to SCHEMA_VERSION. Unknown/older versions
    are merged onto the current defaults, so a field added later simply
    appears with its default rather than erroring."""
    merged = _merge(DEFAULTS, doc or {})
    merged["version"] = SCHEMA_VERSION
    # Legacy persona ids (an older document's dry_jarvis) read as their
    # public id and are persisted canonical at the next write.
    per = (merged.get("dj") or {}).get("persona")
    if isinstance(per, dict) and "profile" in per:
        per["profile"] = dj_persona.canonical_profile_id(per["profile"])
    _resolve_active_output(merged)
    return merged


def _resolve_active_output(doc: dict[str, Any]) -> None:
    """THE ACTIVE DJ SPEAKER MUST BE ONE THE DJ IS ALLOWED ON.

    `outputs.enabled` is the allow-list -- the speakers Parlorwave may DJ
    on -- and `default_request_output` is which one it is following now.
    Nothing tied the second to the first: the active speaker could name a
    player that was never enabled, or one the owner had since removed from
    the list, and a break would be routed to a speaker the DJ was
    explicitly not allowed to use.

    Resolved rather than refused. Removing a speaker from the allow-list is
    an ordinary thing to do, and it must not fail the write that does it:
    an active selection that is no longer permitted is CLEARED, and
    selection then falls back to the first enabled output exactly as it
    does for an installation that has never chosen one.

    Applied on every load and every save, so a document written by an older
    build, or edited by hand, is corrected on the way in rather than
    trusted.
    """
    out = (doc.get("dj") or {}).get("outputs")
    if not isinstance(out, dict):
        return
    active = out.get("default_request_output")
    # ONLY a well-formed selection is resolved. A number, a blank string or
    # any other wrong type is left exactly where it is so validate() still
    # refuses it -- quietly turning a malformed value into None would hand
    # back 200 for a write that was nonsense.
    if not isinstance(active, str) or not active.strip():
        return
    enabled = [o for o in (out.get("enabled") or []) if o]
    if active not in enabled:
        out["default_request_output"] = None


def load(path: Path | None = None) -> dict[str, Any]:
    """Current settings. A missing, empty or unreadable file yields defaults --
    settings must never be able to take the service down."""
    p = _path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        raw = {}
    return _default_malformed_speech(_clamp_stored_cadence(_migrate(raw)))


def _default_malformed_speech(doc: dict[str, Any]) -> dict[str, Any]:
    """A non-boolean dj.speech_enabled, READ FROM DISK, reads as the
    backwards-compatible default (true): only an explicit false silences
    the DJ, and a hand-damaged value must not quietly do it. Applied on
    load and not in _migrate for the same reason as _clamp_stored_cadence:
    a caller WRITING a non-boolean must still be refused by validate(),
    and without this every later save of any OTHER setting would fail
    validation on the damaged value. dj_speech_setting_problem reads the
    raw document and reports the damage until a save replaces it."""
    dj = doc.get("dj")
    if isinstance(dj, dict) and not isinstance(dj.get("speech_enabled"),
                                               bool):
        dj["speech_enabled"] = DEFAULTS["dj"]["speech_enabled"]
    # the story switches: a damaged value reads as its default (preference
    # on, rollout gate OFF)
    for key in ("stories_enabled", "stories_rollout", "atmosphere"):
        if isinstance(dj, dict) and not isinstance(dj.get(key), bool):
            dj[key] = DEFAULTS["dj"][key]
    return doc


def _clamp_stored_cadence(doc: dict[str, Any]) -> dict[str, Any]:
    """A cadence below the one-minute floor, READ FROM DISK, reads as the
    floor.

    The bound used to be `0 < cad`, so a document written then can carry
    0.5 -- and every later save of any OTHER setting validates the merged
    whole and would fail on it. Deliberately applied on load and not in
    _migrate: a caller WRITING half a minute must still be refused, and
    told the range, rather than quietly given a different number."""
    prog = (doc.get("dj") or {}).get("programming")
    if isinstance(prog, dict):
        cad = prog.get("cadence_min_minutes")
        if (isinstance(cad, (int, float)) and not isinstance(cad, bool)
                and 0 < cad < 1):
            prog["cadence_min_minutes"] = 1
    return doc


def _normalize_station_identity(patch: dict[str, Any]) -> None:
    if not isinstance(patch, dict):
        return
    st = patch.get("station")
    if isinstance(st, dict):
        ident = st.get("identity")
        if isinstance(ident, dict) and "name" in ident and isinstance(ident["name"], str):
            raw = ident["name"].strip()
            ident["name"] = DEFAULT_STATION_NAME if not raw else raw


def save(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Merge `patch` onto current settings, validate the RESULT, persist
    atomically, return the stored document.

    Partial updates are the contract: a caller may send only the keys it
    intends to change. Validation always runs against the merged whole, so a
    partial update can never leave an invalid document on disk.
    """
    if patch is not None and not isinstance(patch, dict):
        raise SettingsError("settings payload must be an object")
    patch_copy = deepcopy(patch or {})
    _normalize_station_identity(patch_copy)
    p = _path(path)
    # Announcements are held closed whenever the document on disk cannot
    # be trusted for them -- unreadable, or a malformed announcements
    # container or value (announcements_config's own verdict). A write
    # that does not itself set station.announcements.enabled must not be
    # what turns them back on.
    closed_before = announcements_config(path).get("problem") is not None
    touches_dial = isinstance(patch_copy.get("station"), dict) \
        and "dial" in patch_copy["station"]
    dial_now = dial_config(path) if touches_dial else None
    dial_untrusted = touches_dial and dial_now.get("problem") is not None
    merged = _migrate(_merge(load(path), patch_copy))
    if touches_dial:
        # The Dial block is built on the SANITISED stored block (dial_config
        # reads a damaged one as empty), never deep-merged onto raw stored
        # bytes: a junk key in a damaged block must not make it permanently
        # unsaveable.
        base = {k: dial_now[k] for k in station_intent.DIAL_DEFAULTS}
        patch_dial = patch_copy["station"]["dial"]
        merged["station"]["dial"] = _merge(
            base, patch_dial if isinstance(patch_dial, dict) else {})
        if not isinstance(patch_dial, dict):
            raise SettingsError("station.dial must be an object")
    if closed_before:
        _hold_announcements_closed(patch_copy, merged)
    validate(merged)
    if touches_dial:
        try:
            merged["station"]["dial"] = station_intent.validate_dial(
                merged["station"]["dial"])
        except station_intent.IntentError as exc:
            raise SettingsError(f"station.dial: {exc}") from None
    if (closed_before or dial_untrusted) and p.exists():
        _preserve_untrusted(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a crash mid-write can never leave a truncated document.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".musicbrain_settings.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return merged


def get_section(section: str, path: Path | None = None) -> dict[str, Any]:
    """One section of the document. Raises SettingsError on an unknown name."""
    if section not in SECTIONS:
        raise SettingsError(
            f"unknown settings section: {section} (expected one of "
            f"{', '.join(SECTIONS)})")
    return redacted(load(path))[section]


def save_section(section: str, patch: dict[str, Any],
                 path: Path | None = None, *,
                 generic: bool = False) -> dict[str, Any]:
    """Partial update confined to ONE section, returning that section.

    This is the only write path the API exposes. Confinement is structural
    rather than a filter: the patch is nested under `section` before merging,
    so a caller cannot reach another section however it shapes its payload --
    a `dj` PUT carrying {"session": {...}} writes dj.session, which then fails
    validation, instead of silently changing playback lifecycle.

    `generic` is the API's generic section route (PUT /settings/<section>):
    such a write may send a voice-owned value (VOICE_OWNED_PATHS) back
    unchanged but never change one. voice_library, which owns those values,
    writes without it.
    """
    if section not in SECTIONS:
        raise SettingsError(
            f"unknown settings section: {section} (expected one of "
            f"{', '.join(SECTIONS)})")
    if patch is not None and not isinstance(patch, dict):
        raise SettingsError(f"{section} settings payload must be an object")
    patch = deepcopy(patch or {})
    if section == "station" and "dial" in patch:
        # Preset ids are minted by save_dial_presets alone, so a Dial
        # mapping can trust them; the generic section route may not write
        # the block around it.
        raise SettingsError("station.dial is saved through the Dial "
                            "Stations route, not the station settings")
    _keep_unchanged_secrets(section, patch, load(path))
    _reject_unknown(patch, DEFAULTS[section], (section,))
    if generic:
        _refuse_voice_owned_change(section, patch, path)
    return redacted(save({section: patch}, path))[section]


def dj_allowed_for_queue(queue_id: str, path: Path | None = None) -> bool:
    """Is ParlorWave permitted to insert breaks into this queue?

    The whole group-escalation rule in one line: queue_id is the player_id of
    the output that OWNS the queue, so a grouped child's queue_id is the
    group's id. Enabling a child therefore never authorises its group.

    NOTE: nothing calls this yet -- break scheduling is wired up in a later
    step. It lives here so the rule has exactly one definition.
    """
    if not queue_id:
        return False
    dj = load(path)["dj"]
    return bool(dj.get("enabled")) and queue_id in (dj["outputs"]["enabled"] or [])


def dj_stories_enabled(path: Path | None = None) -> bool:
    """The user's preference: does the listener want biography stories?
    Only an explicit false says no (the product default is on)."""
    try:
        return load(path)["dj"].get("stories_enabled") is not False
    except Exception:
        return True


def dj_stories_rollout(path: Path | None = None) -> bool:
    """The temporary rollout gate. Only an explicit true opens it;
    a missing key, a damaged value or a failed read keeps it shut."""
    try:
        return load(path)["dj"].get("stories_rollout") is True
    except Exception:
        return False


def dj_stories_active(path: Path | None = None) -> bool:
    """Is the music story path LIVE? The preference AND the rollout gate."""
    return dj_stories_rollout(path) and dj_stories_enabled(path)


def dj_atmosphere(path: Path | None = None) -> bool:
    """Atmospheric hosting. Only an explicit true turns it on."""
    try:
        return load(path)["dj"].get("atmosphere") is True
    except Exception:
        return False


def dj_speech_enabled(path: Path | None = None) -> bool:
    """May ordinary DJ hosting be SPOKEN? ONLY AN EXPLICIT false says no.

    Speech is the historical behaviour -- every installation spoke before
    this key existed -- so anything that is not the operator's deliberate
    `false` resolves to it: a missing key (_migrate supplies the default),
    a hand-damaged non-boolean value (load() reads it as the default and
    health reports dj_speech_setting_invalid), or a failed read. Failing
    toward silence is dj_allowed_for_queue's job, which gates the whole DJ
    and still returns False on any settings failure. Deliberately says
    nothing about news or announcements."""
    try:
        return load(path)["dj"].get("speech_enabled") is not False
    except Exception:
        return True


#: /health/problems code: the stored dj.speech_enabled is not a boolean.
DJ_SPEECH_SETTING_INVALID = "dj_speech_setting_invalid"


def dj_speech_setting_problem(path: Path | None = None) -> str | None:
    """DJ_SPEECH_SETTING_INVALID when the document ON DISK carries a
    dj.speech_enabled that is not a boolean (a hand edit validate() never
    saw), else None. load() has already read such a value as the default,
    so the DJ speaks; this only says why the stored value is being
    ignored. A missing or unreadable document is not this problem."""
    status, doc = _document_state(_path(path))
    if status != "ok":
        return None
    dj = doc.get("dj")
    if not isinstance(dj, dict) or "speech_enabled" not in dj:
        return None
    return (None if isinstance(dj["speech_enabled"], bool)
            else DJ_SPEECH_SETTING_INVALID)


def queue_hygiene_permitted(path: Path | None = None) -> bool:
    """Has the operator PERMITTED queue hygiene? Settings failures read as
    not permitted: hygiene must fail toward leaving the queue alone. This is
    the permission half only; the engine's own persistent safety latch is
    the other half, and both must agree (queue_hygiene.py)."""
    try:
        return bool(load(path)["playback"].get("queue_hygiene"))
    except Exception:  # noqa: BLE001 - never a reason to mutate
        return False


def program_audio_as_sound_effect(path: Path | None = None) -> bool:
    """Is ParlorWave's program audio enqueued as a Music Assistant sound
    effect (the prototype)? Settings failures read as False: the proven
    track path."""
    try:
        return load(path)["playback"].get("program_audio_as_sound_effect") is True
    except Exception:  # noqa: BLE001 - the safe answer is the old behaviour
        return False


def _document_state(p: Path) -> tuple[str, Any]:
    """("missing", None) | ("ok", doc) | ("unreadable", None) for the
    document ON DISK -- unlike load(), which turns the last two alike into
    defaults so settings can never take the service down."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "missing", None
    except (OSError, UnicodeDecodeError):
        return "unreadable", None
    try:
        doc = json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        return "unreadable", None
    if not isinstance(doc, dict):
        return "unreadable", None
    return "ok", doc


def _hold_announcements_closed(patch: dict[str, Any],
                               merged: dict[str, Any]) -> None:
    """save() merges onto load(), which reads an untrusted document as
    DEFAULTS (enabled). Unless THIS write sets station.announcements.enabled
    (the operator on the Station card), the result keeps announcements
    off: Start Radio's output intent, a voice save or a station rename must
    never be what turns programming back on. A malformed value is left for
    validate() to refuse (422)."""
    ann_patch = ((patch.get("station") or {}).get("announcements")
                 if isinstance(patch.get("station"), dict) else None)
    if isinstance(ann_patch, dict) and "enabled" in ann_patch:
        return
    station = merged.get("station")
    ann = station.get("announcements") if isinstance(station, dict) else None
    if isinstance(ann, dict):
        ann["enabled"] = False


def _preserve_untrusted(p: Path) -> None:
    """Keep the untrusted document's exact bytes beside it before it is
    replaced, under a name no later copy can overwrite. If that cannot be
    done, the write is REFUSED: the original is never destroyed unseen."""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    for n in range(100):
        dest = p.with_name(f"{p.name}.unreadable-{stamp}"
                           + (f"-{n}" if n else ""))
        try:
            with open(p, "rb") as src, open(dest, "xb") as out:
                shutil.copyfileobj(src, out)
        except FileExistsError:
            continue
        except OSError as exc:
            raise SettingsError(
                "the settings document on disk cannot be read and could not "
                "be preserved, so it will not be overwritten") from exc
        logger.error("settings: the document on disk could not be trusted "
                     "and is being replaced; its bytes were kept in %s, and "
                     "announcements are held OFF until switched on again",
                     dest.name)
        return
    raise SettingsError("could not preserve the untrusted settings document")


ANNOUNCEMENTS_SETTINGS_UNREADABLE = "settings_unreadable"
ANNOUNCEMENTS_SETTINGS_INVALID = "announcements_settings_invalid"


def announcements_config(path: Path | None = None) -> dict[str, Any]:
    """The station.announcements section as the announcement engine reads
    it, in one typed answer, failing CLOSED whenever the operator's intent
    is uncertain.

    load() deliberately turns an unreadable document into DEFAULTS (settings
    must never take the service down) -- and the default is enabled, so
    reading through load() alone would AIR programming on a corrupt file.
    This reader therefore looks at the document itself:

      missing file            a first run: the defaults stand
      unreadable, empty, not  closed, problem=settings_unreadable
      JSON, or not an object
      a malformed value in    closed, problem=announcements_settings_invalid
      station.announcements   (a hand edit validate() never saw)

    `problem` is None when the answer can be trusted."""
    def closed(problem: str) -> dict[str, Any]:
        return {"enabled": False, "smart_rewriting": False,
                "late_air_minutes": 40, "default_voice_id": "",
                "combine_with_dj": False, "combine_wait_minutes": 8,
                "problem": problem}

    status, doc = _document_state(_path(path))
    if status == "unreadable":
        return closed(ANNOUNCEMENTS_SETTINGS_UNREADABLE)
    if status == "missing":
        doc = {}
    try:
        ann = _migrate(doc)["station"]["announcements"]
    except Exception:  # noqa: BLE001 - an unmergeable document
        return closed(ANNOUNCEMENTS_SETTINGS_UNREADABLE)
    if not isinstance(ann, dict):
        return closed(ANNOUNCEMENTS_SETTINGS_INVALID)
    enabled = ann.get("enabled")
    smart = ann.get("smart_rewriting")
    late = ann.get("late_air_minutes")
    dvoice = ann.get("default_voice_id")
    combine = ann.get("combine_with_dj")
    cwait = ann.get("combine_wait_minutes")
    if (not isinstance(enabled, bool) or not isinstance(smart, bool)
            or not isinstance(late, int) or isinstance(late, bool)
            or not 5 <= late <= 120
            or not isinstance(dvoice, str) or len(dvoice) > MAX_VOICE_ID_LEN
            or not isinstance(combine, bool)
            or not isinstance(cwait, int) or isinstance(cwait, bool)
            or not COMBINE_WAIT_MIN <= cwait <= COMBINE_WAIT_MAX):
        return closed(ANNOUNCEMENTS_SETTINGS_INVALID)
    return {"enabled": enabled, "smart_rewriting": smart,
            "late_air_minutes": late, "default_voice_id": dvoice.strip(),
            "combine_with_dj": combine,
            # never longer than the window the occurrence may air in
            "combine_wait_minutes": min(cwait, late),
            "problem": None}


DIAL_SETTINGS_UNREADABLE = "settings_unreadable"
DIAL_SETTINGS_INVALID = "dial_settings_invalid"


def dial_config(path: Path | None = None) -> dict[str, Any]:
    """The saved Dial Stations, read from the document itself, failing
    SAFE: {"version", "next_id", "presets", "problem"}.

      missing file                  a first run: no presets, no problem
      unreadable / not an object    no presets, problem=settings_unreadable
      a malformed station.dial      no presets, problem=dial_settings_invalid

    Safe means EMPTY, never guessed: a Dial position must not start a
    station a damaged document only appears to describe. Nothing else in
    the document is affected -- load() and every other section read as
    they always have."""
    def closed(problem: str, next_id: int = 1) -> dict[str, Any]:
        return dict(deepcopy(station_intent.DIAL_DEFAULTS),
                    next_id=next_id, problem=problem)

    status, doc = _document_state(_path(path))
    if status == "unreadable":
        return closed(DIAL_SETTINGS_UNREADABLE)
    if status == "missing":
        doc = {}
    try:
        block = _migrate(doc)["station"]["dial"]
    except Exception:  # noqa: BLE001 - an unmergeable document
        return closed(DIAL_SETTINGS_UNREADABLE)
    try:
        return dict(station_intent.validate_dial(block), problem=None)
    except station_intent.IntentError:
        return closed(DIAL_SETTINGS_INVALID, _salvage_next_id(block))


def _salvage_next_id(block: Any) -> int:
    """For a damaged Dial block: an id counter that is still past every id
    the damaged list mentions, so starting over can never re-issue one."""
    best = 0
    presets = block.get("presets") if isinstance(block, dict) else None
    for p in presets if isinstance(presets, list) else []:
        pid = p.get("id") if isinstance(p, dict) else None
        if isinstance(pid, str) and pid.startswith("preset_") \
                and pid[7:].isdigit():
            best = max(best, int(pid[7:]))
    nid = block.get("next_id") if isinstance(block, dict) else None
    if isinstance(nid, int) and not isinstance(nid, bool):
        best = max(best, nid - 1)
    return min(best + 1, 999999)


def save_dial_presets(presets: Any, path: Path | None = None
                      ) -> dict[str, Any]:
    """Replace the Dial's preset list (new presets are minted ids). Only
    station.dial is written. A damaged stored block is replaced -- its
    bytes kept beside the document first -- and its ids are never reused.
    Raises SettingsError."""
    current = dial_config(path)
    base = {k: current[k] for k in station_intent.DIAL_DEFAULTS}
    try:
        block = station_intent.apply_presets(base, presets)
    except station_intent.IntentError as exc:
        raise SettingsError(str(exc)) from None
    save({"station": {"dial": block}}, path)
    return dial_config(path)


def resolved_station_name(settings: dict[str, Any] | None = None) -> str:
    """The authoritative effective on-air station name.

    Owns default fallback, trimming, sanitization, and length policy.
    All runtime consumers must use this one resolver.
    """
    if settings is None:
        try:
            settings = load()
        except Exception:
            return DEFAULT_STATION_NAME
    raw = None
    if isinstance(settings, dict):
        st = settings.get("station")
        if isinstance(st, dict):
            ident = st.get("identity")
            if isinstance(ident, dict):
                raw = ident.get("name")
            elif "name" in st:
                raw = st.get("name")
    if raw is None or not isinstance(raw, str):
        return DEFAULT_STATION_NAME
    name = raw.strip()
    if not name:
        return DEFAULT_STATION_NAME
    # Strip control characters (ord < 32 or unicode category starting with C)
    cleaned = "".join(c for c in name if ord(c) >= 32 and not unicodedata.category(c).startswith("C"))
    # Strip any markup tags (<...>)
    if "<" in cleaned or ">" in cleaned:
        cleaned = re.sub(r"<[^>]*>", "", cleaned).strip()
    if not cleaned:
        return DEFAULT_STATION_NAME
    return cleaned[:MAX_STATION_NAME_LENGTH]

