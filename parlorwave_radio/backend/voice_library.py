"""The DJ voice library: who the DJ sounds like.

Voice is WHO; personality is HOW. Selecting a voice writes ONLY
dj.persona.voice (the logical id + the Chatterbox asset it resolves to);
personality, chattiness, station feel, programming, output and transport
are untouched by construction -- the write path never reaches them.

Two shelves:

  KOKORO    the seven curated built-in DJ voices. Their identity lives in
            kokoro_voices; this module only shelves them, so the speaker
            numbers behind them never reach a customer surface.
  CUSTOM    three stable slots (custom_1..3). Identity is the SLOT id;
            the display name is a setting the user can change freely; the
            Chatterbox asset name is fixed per slot (jarvis_custom_N) so
            renames never touch files and display names never become
            paths. A slot without an uploaded sample is offered as empty.

Uploads go through the speech service's own documented store (POST
/voices multipart) -- no second sample store, no filesystem paths in any
API, no shell. Validation here: real audio magic (WAV/MP3/FLAC), bounded
size, safe fixed asset names.

Preview synthesizes one short phrase -- the fixed line, or a short one
the owner typed -- and returns the audio bytes to the caller. It never
touches Music Assistant, never creates a break, never enters history or
cadence.

ENROLLMENT. A custom voice is a recording
of a person, so nothing stores one without the owner's affirmative
acknowledgement that they may create and use it (CONSENT_TEXT, versioned
as CONSENT_VERSION). create_voice, replace_voice and upload_sample all
refuse without it, and the acknowledgement is recorded with the slot
(dj.voice_slots.<slot>.authorization) and cleared with it. What the voice
may then be USED for is the scope (dj.voice_scopes), chosen explicitly at
enrollment -- never inherited from whoever held the slot before. There is
deliberately no way in from a URL, a path or anything but the recording's
own bytes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("musicbrain")

# There is deliberately NO Chatterbox "built-in" shelf: a cloned voice is
# always someone's own recording, so it lives in a custom slot like any
# other; presenting one as a product built-in would be a lie in the UI.
CUSTOM_SLOTS = ("custom_1", "custom_2", "custom_3")
# The FIXED Chatterbox asset name per slot. Display names never appear in
# file or asset names; replacing a sample re-uses the same asset name.
CUSTOM_ASSET = {slot: f"jarvis_{slot}" for slot in CUSTOM_SLOTS}

#: What a break speaks with when nothing is selected: the speech service's
#: own default sample. This is the pre-library behaviour and the only safe
#: landing after a selected custom sample is deleted -- never a curated
#: voice the customer did not pick.
SERVICE_DEFAULT_ID = "default_voice_1"
SERVICE_DEFAULT_NAME = "Service default"

PREVIEW_TIMEOUT_S = 30.0
#: The longest line an owner may type for a preview. A preview is a
#: sentence or two, heard in the browser; it is not a way to render audio.
PREVIEW_MAX_CHARS = 200

MAX_SAMPLE_BYTES = 10 * 1024 * 1024
MIN_SAMPLE_BYTES = 16 * 1024
MAX_NAME_LEN = 40
#: a remembering-aid, not a description: "calm, slower reader"
MAX_NOTE_LEN = 80
#: The preview line: a host's own
#: sentence, with a comma list, a question and a short close -- the pauses
#: and lifts a voice has to carry, in words a DJ would actually say. The
#: old line ("Hello, this is your DJ test voice. Commas, pauses...") read
#: as a test pattern and was clipped in the box that showed it.
PREVIEW_PHRASE = ("You're listening to ParlorWave Radio. There's more great "
                  "music ahead, a few surprises along the way, and plenty "
                  "worth sticking around for. Ready? Let's get back to it.")
#: The one sentence this product may say about where a custom voice is
#: processed. It is a statement of FACT -- the recording goes to whatever
#: voice engine the owner configured -- and deliberately not a privacy
#: claim: a provider on a private address is still a provider, and no
#: hostname or IP heuristic can tell us it is local. If the engine ever
#: reports a trustworthy "local" capability, THAT is what may change this
#: line, and nothing else.
PROCESSING_NOTE = ("Custom voices are processed by your configured "
                   "ParlorWave voice engine.")
#: What the Booth shows in place of upload controls when no engine exists.
CLONING_UNCONFIGURED = (
    "Create a DJ voice from your own clean recording. Requires the "
    "optional ParlorWave Voice Engine.")
CLONING_UNREACHABLE = (
    "The ParlorWave Voice Engine is configured but unreachable right now, "
    "so custom voices cannot be created or previewed.")
#: Whether NEW enrollment is offered: making a voice, putting a new
#: recording under one, or storing a sample for a slot. False refuses all
#: three (VoiceEnrollmentUnavailable) before anything is looked at or
#: uploaded, whatever engine is configured; voices that already exist are
#: untouched either way. True for the supported engine, Chatterbox TTS
#: API 2.1.0, whose whole path works (engine found in Settings, a replace
#: that survives the engine's 409, create -> preview -> select -> DJ render
#: -> restart, end to end).
ENROLLMENT_QUALIFIED = True
ENROLLMENT_UNQUALIFIED_MESSAGE = (
    "Making a voice from a recording is not available in this release "
    "while it is still being tested. Voices already on this installation "
    "keep working.")
#: What a mutation says when there is no engine to mutate anything on.
NO_ENGINE_MESSAGE = (
    "Custom voices need the optional ParlorWave Voice Engine, which this "
    "installation has not got. Home Assistant and built-in DJ voices are "
    "unaffected.")
ENGINE_FAILED_MESSAGE = (
    "The ParlorWave Voice Engine could not complete that. Check that it is "
    "running, then try again.")

#: The acknowledgement an enrollment requires, word for word as a UI
#: shows it, and the version a client sends back to say it showed THIS
#: text. A changed sentence is a new version: an old client's "yes" was
#: to a sentence it no longer matches, and is refused.
CONSENT_TEXT = "I have permission to create and use this voice in Parlorwave."
CONSENT_VERSION = "voice-permission-v1"
AUTHORIZATION_GRANTED = "granted"
AUTHORIZATION_REQUIRED_MESSAGE = (
    "Confirm that you have permission to create and use this voice before "
    "it is saved.")
#: How a sample reached us -- recorded in the Booth or chosen as a file.
#: Stored beside the acknowledgement; says nothing about the speaker.
SAMPLE_SOURCES = ("recording", "file")

#: What the shelf lets the owner tune on a built-in voice. 1.0 = the
#: engine's own; the sidecar accepts wider, the shelf offers taste.
SPEED_RANGE = (0.7, 1.3)
PAUSE_RANGE = (0.5, 2.5)


class VoiceError(ValueError):
    """A refusal with a human reason (mapped to HTTP 4xx by the API)."""


class VoiceEngineUnavailable(VoiceError):
    """Custom voices need the optional voice engine, and this installation
    has not got one right now. A DISTINCT type because it is not the
    caller's mistake: the API answers 503 with a settled sentence, never a
    provider's own exception text, a URL or an environment variable name.
    Home Assistant and the curated built-in voices are unaffected."""

    code = "voice_engine_unavailable"


class VoiceEngineFailed(VoiceError):
    """The voice engine was there and the operation still failed. Same
    posture as above: the customer gets a sentence, the log gets the
    detail."""

    code = "voice_engine_failed"


class VoiceSlotsFull(VoiceError):
    """Every custom voice this release can hold is already in use."""

    code = "voice_slots_full"


class VoiceEnrollmentUnavailable(VoiceError):
    """New enrollment is not offered in this release (ENROLLMENT_QUALIFIED).
    Not the caller's mistake and not a fault: the API answers 403 with the
    settled sentence."""

    code = "voice_enrollment_unavailable"


class VoiceCannotSpeak(VoiceError):
    """A user chose a voice as the DJ and it could not produce speech
    when asked to (speak_check). The selection is NOT written: a voice is
    never reported as the DJ while synthesis cannot use it."""

    code = "voice_cannot_speak"


class VoiceAuthorizationRequired(VoiceError):
    """An enrollment arrived without the owner's acknowledgement that they
    may create and use the voice (or with one to a different text)."""

    code = "voice_authorization_required"


class VoiceNameTaken(VoiceError):
    """Another voice on the shelf already answers to that display name."""

    code = "voice_name_taken"


class VoiceNotAuthorized(VoiceError):
    """The voice is not allowed for what was asked of it. Distinct so a UI
    can say "allow it first" rather than "something went wrong"."""

    code = "voice_not_authorized"


def _display_name(voice_id: str, asset: str) -> str:
    """A customer-readable name for a voice this shelf did not define."""
    import tts_voices
    if asset:
        return tts_voices.friendly_name(asset)
    return voice_id.replace("_", " ").title()


def _selected_provider(refs: dict[str, Any]) -> str:
    """Which provider the ROUTER would use for the stored refs. Derived,
    never stored: the registry decides providers, not a saved string."""
    import tts_router
    try:
        return str(tts_router.decide(dict(refs)).get("provider") or "")
    except Exception:                       # noqa: BLE001
        return ""


def _kokoro_shelf() -> list[dict[str, Any]]:
    """The seven curated built-ins, with live availability. A Kokoro that
    is switched off or unreachable makes them UNAVAILABLE, never absent:
    a voice that vanishes from the shelf looks like a deleted voice."""
    import kokoro_voices
    import tts_router
    try:
        health = tts_router.health()
        available = bool(health.get("ok"))
    except Exception:                       # noqa: BLE001 - never block the shelf
        available = False
    return [{"voice_id": v["voice_id"], "name": v["label"],
             "provider": kokoro_voices.PROVIDER, "available": available}
            for v in kokoro_voices.shelf()]


def _is_kokoro(voice_id: str) -> bool:
    import kokoro_voices
    return kokoro_voices.is_builtin(voice_id)


# --- Home Assistant voices ------------------------------------------------
# A Home Assistant TTS entity (tts.piper, tts.home_assistant_cloud, ...) is
# a voice on this shelf under the logical id "ha_" + its entity id with the
# dot as an underscore: tts.piper <-> ha_tts_piper. The mapping is a pure
# function of the id, so a selection stays valid whether or not Home
# Assistant can be reached at the moment the shelf is read. Finer voice
# selection inside a provider is provider-specific; for now the ENTITY is
# the voice, and the integration's onboarding will populate the list.
HA_PROVIDER = "home_assistant"
HA_VOICE_PREFIX = "ha_"
_HA_VOICE_RE = re.compile(r"^ha_tts_[a-z0-9_]+$")


def _is_ha(voice_id: str) -> bool:
    return bool(_HA_VOICE_RE.match(voice_id or ""))


def ha_voice_id(entity_id: str) -> str:
    return HA_VOICE_PREFIX + entity_id.replace(".", "_", 1)


def ha_entity_for(voice_id: str) -> str:
    return "tts." + voice_id[len(HA_VOICE_PREFIX) + len("tts_"):]


def _ha_shelf() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(voices, status). Voices come from Home Assistant's own tts.*
    entities; an unreachable Home Assistant yields an EMPTY list with an
    honest status, never an invented one. Never raises."""
    import ha_tts
    import providers
    if not providers.ha_configured():
        return [], {"configured": False, "ok": False, "error": ""}
    try:
        entities = ha_tts.list_entities()
    except Exception as exc:  # noqa: BLE001 - never block the shelf
        return [], {"configured": True, "ok": False, "error": str(exc)[:160]}
    voices = [{"voice_id": ha_voice_id(e["entity_id"]), "name": e["name"],
               "provider": HA_PROVIDER, "entity_id": e["entity_id"],
               "available": bool(e.get("available"))}
              for e in entities if _is_ha(ha_voice_id(e["entity_id"]))]
    return voices, {"configured": True, "ok": True, "error": ""}


def _is_shelved(voice_id: str) -> bool:
    return _is_kokoro(voice_id) or voice_id in CUSTOM_SLOTS or _is_ha(voice_id)


#: The kind a custom slot resolves to (kind_of); the other two kinds are
#: their provider names.
CUSTOM_KIND = "custom"


def kind_of(voice_id: str) -> str | None:
    """Which shelf an id belongs to, from the id alone (no I/O): the
    Home Assistant provider, the Kokoro provider, CUSTOM_KIND, or None for
    an id no shelf claims. The one function every "is this a voice we
    know" question routes through."""
    if not isinstance(voice_id, str) or not voice_id:
        return None
    if _is_ha(voice_id):
        return HA_PROVIDER
    if _is_kokoro(voice_id):
        import kokoro_voices
        return kokoro_voices.PROVIDER
    if voice_id in CUSTOM_SLOTS:
        return CUSTOM_KIND
    return None


# --- what a voice is authorized for -----------------------------------------
# Voice is WHO; scope is WHOSE PROGRAMMING may use it. Every shelved voice
# has always been selectable as the DJ voice -- that is scope "dj", the
# meaning of an absent entry, and nothing here changes it. What is new is
# the ANNOUNCEMENT scope: a saved voice a calendar event may name for one
# announcement. That is an explicit grant the owner makes per voice; it is
# never inferred from the voice existing, from the voice being the DJ's,
# or from anything a calendar event says. "both" grants both.
SCOPE_DJ = "dj"
SCOPE_ANNOUNCEMENT = "announcement"
SCOPE_BOTH = "both"
ANNOUNCEMENT_SCOPES = (SCOPE_ANNOUNCEMENT, SCOPE_BOTH)
DJ_SCOPES = (SCOPE_DJ, SCOPE_BOTH)

#: Why a calendar's voice request was not honoured -- recorded, never
#: guessed around. Distinct from render-time unavailability (an
#: AUTHORIZED voice that cannot speak right now).
REJECT_UNKNOWN = "voice_unknown"
REJECT_NOT_AUTHORIZED = "voice_not_authorized"
#: More than one saved voice answers to the requested display name. A
#: name is a courtesy; when it stops being unique nothing is chosen.
REJECT_AMBIGUOUS = "voice_ambiguous"

# --- render-time revalidation ----------------------------------------------
# An occurrence's voice is decided when it is committed and spoken when it
# is rendered. Whatever the gap, a grant made at commit must not outlive
# the registry it was made against: before synthesis the committed id is
# checked again, against the registry AS IT IS NOW, and any of these
# refuses it (the announcement then falls to the next voice down, its text
# untouched, the reason recorded).
CHECK_MISSING = "voice_missing"                     # no such voice any more
CHECK_WITHDRAWN = "voice_authorization_withdrawn"  # the grant was withdrawn
CHECK_IDENTITY_CHANGED = "voice_identity_changed"   # same slot, another voice
CHECK_REASONS = (CHECK_MISSING, CHECK_WITHDRAWN, CHECK_IDENTITY_CHANGED)

#: A custom slot's IDENTITY: the full sha256 hex digest of the sample
#: bytes stored in it (dj.voice_slots.<slot>.identity), minted by
#: upload_sample and cleared by delete_sample. A slot id is reusable --
#: custom_1 may be Evening Reader today and a stranger next week -- so a
#: decision made about "custom_1" carries this fingerprint, and a render
#: compares it with the slot's current one. It is a fingerprint of WHICH recording,
#: never the recording: no audio is snapshotted anywhere. A curated or
#: Home Assistant voice is its own identity (its id never changes hands).
#: The digest is kept WHOLE: this value takes part in an authorization
#: decision, not in de-duplication, and nothing stores it but settings.
IDENTITY_HEX = 64


def sample_identity(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def identity_of(voice_id: str) -> str | None:
    """The identity a decision about `voice_id` should carry: the slot's
    current sample fingerprint for a custom voice (None when the slot has
    none recorded -- empty, or filled before fingerprints existed), the id
    itself for a curated or Home Assistant voice, None for no voice."""
    kind = kind_of(voice_id)
    if kind is None:
        return None
    if kind == CUSTOM_KIND:
        return str((_slots().get(voice_id) or {}).get("identity") or "") or None
    return voice_id


def check_announcement_voice(voice_id: str, expected_identity: str | None
                             ) -> str | None:
    """Is the voice a commit decided on still the voice, and still
    allowed, RIGHT NOW? ONE contract for every explicitly selected
    announcement voice -- the event's request and the configured default
    alike; only the station's DJ floor, which is not a selected
    announcement voice, is never asked. None when every check passes;
    otherwise the first CHECK_* reason that fails, in this order:

      exists    the id is still a shelved voice (CHECK_MISSING); for a
                custom slot whose commit-time fingerprint is gone, the
                sample was deleted (CHECK_MISSING);
      identity  a custom slot's current fingerprint equals the one the
                commit recorded (CHECK_IDENTITY_CHANGED) -- a slot that
                had no fingerprint then and has one now was re-enrolled;
                one without a fingerprint both times cannot be told apart
                and passes (the synthesizer still proves a sample exists);
      scope     the CURRENT scope permits announcements
                (CHECK_WITHDRAWN). dj.voice_scopes is what a voice may
                do; station.announcements.default_voice_id is only which
                authorized voice is preferred, and grants nothing.

    Pure registry read; never writes, never reaches an engine."""
    kind = kind_of(voice_id)
    if kind is None:
        return CHECK_MISSING
    if kind == CUSTOM_KIND:
        current = identity_of(voice_id)
        expected = expected_identity or None
        if expected and not current:
            return CHECK_MISSING
        if (expected or current) and expected != current:
            return CHECK_IDENTITY_CHANGED
    if scope_of(voice_id) not in ANNOUNCEMENT_SCOPES:
        return CHECK_WITHDRAWN
    return None


def _scopes() -> dict[str, str]:
    import settings
    raw = settings.load()["dj"].get("voice_scopes") or {}
    return {k: v for k, v in raw.items()
            if isinstance(v, str) and v in settings.VOICE_SCOPES}


def scope_of(voice_id: str) -> str:
    return _scopes().get(voice_id) or SCOPE_DJ


def set_scope(voice_id: str, scope: Any) -> dict[str, Any]:
    """Grant or withdraw a voice's uses. Only a shelved voice can carry a
    scope, so an authorization can never be written for a voice that does
    not exist. The DJ's current voice cannot be made announcement-only:
    the persona would be left speaking with a voice it is not allowed --
    pick another DJ voice first, or grant both."""
    import settings
    if not _is_shelved(voice_id):
        raise VoiceError(f"unknown voice: {voice_id}")
    _check_scope(voice_id, scope)
    settings.save_section("dj", {"voice_scopes": {voice_id: scope}})
    return {"voice_id": voice_id, "scope": scope}


def _check_scope(voice_id: str, scope: Any) -> None:
    """The two rules every scope write obeys, wherever it comes from: a
    known scope, and never announcement-only for the voice the DJ is
    speaking with right now."""
    import settings
    if scope not in settings.VOICE_SCOPES:
        raise VoiceError("scope must be one of "
                         f"{', '.join(settings.VOICE_SCOPES)}")
    if scope == SCOPE_ANNOUNCEMENT and (_selected().get("id") or "") == voice_id:
        raise VoiceError("That voice is the DJ voice. Choose another DJ "
                         "voice first, or allow it for both.")


def announcement_default() -> dict[str, Any]:
    """station.announcements.default_voice_id as a UI needs it: the id,
    and whether that voice is allowed for announcements RIGHT NOW. A
    default whose grant was withdrawn stays configured (the render refuses
    it and says so); a UI shows that rather than hiding it. Never raises."""
    import settings
    try:
        vid = str(settings.announcements_config().get("default_voice_id") or "")
    except Exception:  # noqa: BLE001 - a read for display only
        vid = ""
    return {"voice_id": vid or None,
            "allowed": bool(vid) and kind_of(vid) is not None
            and scope_of(vid) in ANNOUNCEMENT_SCOPES}


def set_announcement_default(voice_id: Any, fetch_fn=None) -> dict[str, Any]:
    """Choose which AUTHORIZED voice reads an announcement whose event
    names none; "" puts the station's DJ voice back. Writes
    station.announcements.default_voice_id and nothing else.

    It is a PREFERENCE, never a grant: a voice that is not scoped for
    announcements is refused here (and, should its grant later be
    withdrawn, still refused at render). Nothing in this function writes a
    scope."""
    import settings
    voice_id = str(voice_id or "").strip()
    if voice_id:
        if kind_of(voice_id) is None:
            raise VoiceError(f"unknown voice: {voice_id}")
        if scope_of(voice_id) not in ANNOUNCEMENT_SCOPES:
            raise VoiceNotAuthorized(
                "That voice is not allowed for announcements. Allow it for "
                "announcements first; choosing a default never does that.")
        if (kind_of(voice_id) == CUSTOM_KIND and not identity_of(voice_id)
                and not _slot_has_sample(voice_id, fetch_fn)):
            raise VoiceError("That voice has no recording stored yet.")
    settings.save_section("station", {"announcements": {
        "default_voice_id": voice_id}})
    return {"voice_id": voice_id or None,
            "name": _name_of(voice_id) if voice_id else None}


def _name_of(voice_id: str) -> str:
    return next((n for vid, n in _shelf_names() if vid == voice_id), "")


def _shelf_names() -> list[tuple[str, str]]:
    """(voice_id, display name) for every shelved voice, as the shelf
    shows them: the owner's rename first, the curated name otherwise.
    Home Assistant voices need Home Assistant to be listed; when it cannot
    be, a renamed one is still known by its name (the rename is ours)."""
    import kokoro_voices
    names, slots = _names(), _slots()
    out: list[tuple[str, str]] = []
    for v in kokoro_voices.shelf():
        out.append((v["voice_id"], names.get(v["voice_id"]) or v["label"]))
    for slot in CUSTOM_SLOTS:
        out.append((slot, _custom_name(slot, slots.get(slot) or {})))
    ha_voices, _st = _ha_shelf()
    seen = {vid for vid, _n in out}
    for v in ha_voices:
        out.append((v["voice_id"], names.get(v["voice_id"]) or v["name"]))
        seen.add(v["voice_id"])
    for vid, name in names.items():
        if _is_ha(vid) and vid not in seen and name:
            out.append((vid, name))
    return out


def resolve_announcement_voice(request: Any) -> dict[str, Any]:
    """The voice a calendar event asked for, checked against the registry.

    {"requested": str|None, "voice_id": str|None, "reason": str|None}.
    The request may be a voice id (`custom_1`, `lewis`, `ha_tts_piper`) or
    the voice's display name as the shelf shows it ("Evening Reader"); both
    compare case-insensitively, and names are unique on the shelf by
    construction.
    It is NEVER an asset name, an engine id, a file, a path, a URL or an
    entity id -- none of those are shelved ids, so none of them resolve.
    A match is then honoured only if the owner scoped that voice for
    announcements. Nothing here touches settings or the engine: it reads
    the shelf and answers."""
    import announce_copy
    requested = announce_copy.normalize_voice_request(request)
    out: dict[str, Any] = {"requested": requested, "voice_id": None,
                           "reason": None}
    if not requested:
        return out
    key = requested.casefold()
    voice_id = None
    if _is_shelved(key):
        voice_id = key                      # an id is exact: never ambiguous
    else:
        # A display name is a courtesy. rename()/create_voice() refuse a
        # name already on the shelf, but a Home Assistant voice's name is
        # Home Assistant's and can collide with anything at any time --
        # so EVERY match is collected, and two distinct voices answering
        # to one name is a refusal, never a pick.
        matches = sorted({vid for vid, name in _shelf_names()
                          if name.casefold() == key})
        if len(matches) > 1:
            out["reason"] = REJECT_AMBIGUOUS
            return out
        voice_id = matches[0] if matches else None
    if voice_id is None:
        out["reason"] = REJECT_UNKNOWN
        return out
    if scope_of(voice_id) not in ANNOUNCEMENT_SCOPES:
        out["reason"] = REJECT_NOT_AUTHORIZED
        return out
    out["voice_id"] = voice_id
    return out


def announcement_voices() -> list[dict[str, Any]]:
    """The voices a calendar event may currently name, for a UI or a
    status read. Never raises."""
    try:
        scopes = _scopes()
        allowed = {vid for vid, sc in scopes.items() if sc in ANNOUNCEMENT_SCOPES}
        if not allowed:
            return []
        return [{"voice_id": vid, "name": name, "scope": scopes[vid]}
                for vid, name in _shelf_names() if vid in allowed]
    except Exception:  # noqa: BLE001 - a read for display only
        logger.debug("announcement voice list failed", exc_info=True)
        return []


# --- the DJ on the air -------------------------------------------------------
# dj.persona.voice is the ONE current-DJ setting and select() its one
# writer; "Switch DJ" is select() with the next voice in the lineup. What
# nothing asked before is whether the stored selection is STILL a voice
# the DJ may speak with when a break is rendered. A custom slot is
# reusable and a scope can change after the pick, so the render asks the
# registry again -- the three questions an announcement asks, with DJ
# scope in place of announcement scope -- and a selection that fails
# speaks with the service default instead (the landing a deleted DJ voice
# already had). Nothing is written when that happens: the selection stays
# what the user chose, the library says it is not in force and why, and
# choosing again is the owner's act, never the render's.

#: refs key binding refs.chatterbox_voice to WHICH recording it named when
#: it was chosen: the slot's identity (sample_identity) at selection. In
#: refs because refs is the schema's provider-shaped extensible map and
#: this describes the Chatterbox ref beside it -- including when that ref
#: is only the fallback kept under a Kokoro or Home Assistant voice.
IDENTITY_REF = "chatterbox_identity"
NO_OTHER_DJ_MESSAGE = "No other voice is allowed to be the DJ right now."


def _slot_for_asset(asset: Any) -> str | None:
    return next((s for s, a in CUSTOM_ASSET.items() if a == asset), None)


def check_dj_voice(voice_id: str, expected_identity: str | None
                   ) -> str | None:
    """May `voice_id` speak for the DJ RIGHT NOW, as the recording it was
    chosen as? None, or the first CHECK_* reason that fails -- in the
    order check_announcement_voice uses:

      exists    a custom slot whose recorded fingerprint is gone
                (CHECK_MISSING);
      identity  a custom slot now holds another recording than the one
                selected (CHECK_IDENTITY_CHANGED); no fingerprint either
                time cannot be told apart and passes;
      scope     the CURRENT scope still permits the DJ (CHECK_WITHDRAWN).

    A voice no shelf claims -- the service default, or an enrolled voice
    from before the shelves -- has no scope or fingerprint to ask about
    and passes: it is reported as itself, as it always was. Pure registry
    read; never writes, never reaches an engine."""
    kind = kind_of(voice_id)
    if kind is None:
        return None
    if kind == CUSTOM_KIND:
        current = identity_of(voice_id)
        expected = expected_identity or None
        if expected and not current:
            return CHECK_MISSING
        if (expected or current) and expected != current:
            return CHECK_IDENTITY_CHANGED
    if scope_of(voice_id) not in DJ_SCOPES:
        return CHECK_WITHDRAWN
    return None


#: How long the automatic DJ voice is reused before its providers are
#: asked again: it is read on every render and every status poll.
AUTOMATIC_TTL_S = 60.0
_automatic: dict[str, Any] = {"at": None, "value": None}


def _any_voice_ref(refs: dict[str, Any] | None) -> bool:
    """Has anybody chosen a voice? Any provider's ref says so -- the same
    rule health.voice_selected applies."""
    import tts_router
    return any(str((refs or {}).get(k) or "").strip()
               for k in (tts_router.HA_REF, tts_router.KOKORO_REF,
                         tts_router.CHATTERBOX_REF))


def automatic_dj_voice(now: float | None = None) -> dict[str, Any] | None:
    """The voice the DJ speaks with while NOBODY HAS CHOSEN one, so a fresh
    installation needs no voice-selection step -- or None.

    In order:
      1. None when a voice engine is configured: its own default sample
         speaks -- the route every such installation has always had, kept
         exactly (nothing changes where the old default could speak);
      2. the first curated built-in, when the Kokoro sidecar is on and
         answering;
      3. the first available Home Assistant voice, in the shelf's order.

    {"voice_id", "refs"} -- refs exactly as select() would write them.
    Nothing is written: the moment the owner chooses, that choice is the DJ.
    Cached AUTOMATIC_TTL_S. Never raises."""
    import time
    t = time.monotonic() if now is None else now
    if _automatic["at"] is not None and t - _automatic["at"] < AUTOMATIC_TTL_S:
        return _automatic["value"]
    try:
        value = _automatic_choice()
    except Exception:  # noqa: BLE001 - never block a render or a status
        value = None
    _automatic.update(at=t, value=value)
    return value


def reset_automatic_cache() -> None:
    _automatic.update(at=None, value=None)


def _automatic_choice() -> dict[str, Any] | None:
    import kokoro_voices
    import providers
    import tts_router
    if cloning_configured():
        return None
    if providers.kokoro_enabled() and tts_router.health().get("ok"):
        first = kokoro_voices.shelf()[0]["voice_id"]
        return {"voice_id": first, "refs": {tts_router.KOKORO_REF: first}}
    voices, _status = _ha_shelf()
    for v in voices:
        if v.get("available"):
            return {"voice_id": v["voice_id"],
                    "refs": {tts_router.HA_REF: v["entity_id"]}}
    return None


#: What speak_check says, after the voice's name, for each reason
#: tts_router.synthesize_voice gives. Never a provider's own words.
SPEAK_CHECK_TEXT = "Testing, one, two."
_CANNOT_SPEAK = {
    "home_assistant_unconfigured":
        "Home Assistant's voices cannot be reached from ParlorWave Radio "
        "right now",
    "kokoro_disabled":
        "the built-in voices need the optional voice sidecar, which is not "
        "running",
    "voice_engine_unavailable": "the ParlorWave Voice Engine is not answering",
    "no_sample": "its recording is no longer on the voice engine",
    "provider_error": "its speech service failed when asked to speak",
    "empty_audio": "its speech service returned no audio",
}


#: How speak_check's refusal reads for each owner act that puts a voice
#: on the air ({name}, {why}).
REFUSED_AS_DJ = "{name} was not made the DJ: {why}. The DJ is unchanged."
REFUSED_FOR_SHIFT = ("{name} was not given a DJ shift: {why}. The shifts are "
                     "unchanged.")


def speak_check(voice_id: str, synth_fn=None, *,
                refused: str = REFUSED_AS_DJ) -> None:
    """Prove a voice can speak BEFORE an owner's act puts it on the air --
    choosing it as the DJ, Switch DJ moving to it, or assigning it a DJ
    shift: one short phrase through that voice alone
    (tts_router.synthesize_voice -- no fallback), nothing played, nothing
    stored. VoiceCannotSpeak, with the reason in words (`refused`), when it
    cannot.

    An id no shelf claims is left to select(), which refuses or keeps it
    in its own words; re-picking the voice already in force is a no-op
    there, so there is nothing to prove."""
    import tts_router
    if kind_of(voice_id) is None:
        return
    cur = _selected()
    if voice_id == (cur.get("id") or "") and check_dj_voice(
            voice_id, (cur.get("refs") or {}).get(IDENTITY_REF)) is None:
        return
    fn = synth_fn or tts_router.synthesize_voice
    try:
        fn(SPEAK_CHECK_TEXT, voice_id)
    except tts_router.VoiceUnavailable as exc:
        why = _CANNOT_SPEAK.get(getattr(exc, "reason", ""),
                                "it could not speak just now")
        name = _name_of(voice_id) or "That voice"
        raise VoiceCannotSpeak(refused.format(name=name, why=why)) from exc


def dj_voice_in_force() -> dict[str, Any]:
    """The voice the NEXT DJ render speaks with, decided now.

      voice_id           what speaks: the selection, or SERVICE_DEFAULT_ID
                         when the selection fails check_dj_voice
      selected_voice_id  what the user chose (dj.persona.voice.id)
      refs               the refs to route (tts_router); {} for the
                         service default
      reason             why the selection is not in force, or None
      fallback_withheld  why the Chatterbox fallback kept under a Kokoro
                         or Home Assistant voice was dropped, or None --
                         that fallback can name a custom recording too,
                         and is held to the same checks

    When NOBODY HAS CHOSEN a voice (no ref of any provider -- a fresh
    installation), the result also carries `automatic: True` and speaks
    with automatic_dj_voice() when there is one: that is the only case
    that reads a provider, through a short cache.

    Otherwise a pure registry read: no engine, no network, no write."""
    sel = _selected()
    vid = str(sel.get("id") or "")
    refs = dict(sel.get("refs") or {})
    out: dict[str, Any] = {"voice_id": vid, "selected_voice_id": vid,
                           "refs": refs, "reason": None,
                           "fallback_withheld": None}
    if not _any_voice_ref(refs) and vid in ("", SERVICE_DEFAULT_ID):
        auto = automatic_dj_voice()
        if auto:
            out.update(voice_id=auto["voice_id"], refs=dict(auto["refs"]),
                       automatic=True)
            return out
    reason = check_dj_voice(vid, refs.get(IDENTITY_REF))
    if reason:
        out.update(voice_id=SERVICE_DEFAULT_ID, refs={}, reason=reason)
        return out
    slot = _slot_for_asset(refs.get("chatterbox_voice"))
    if slot and slot != vid:
        withheld = check_dj_voice(slot, refs.get(IDENTITY_REF))
        if withheld:
            refs["chatterbox_voice"] = ""
            out["fallback_withheld"] = withheld
    return out


def _dj_lineup(lib: dict[str, Any]) -> list[dict[str, Any]]:
    """Every shelved voice in rotation order -- the curated built-ins in
    their curated order, the custom voices by slot, then Home Assistant's
    by entity id (the order library() lists them in) -- each with whether
    it may be picked as the DJ right now: allowed for the DJ, and able to
    speak (a live provider; a custom slot the engine lists)."""
    rows = [(k, bool(k.get("available"))) for k in lib.get("kokoro") or []]
    rows += [(c, c.get("has_sample") is True)
             for c in lib.get("customs") or []]
    rows += [(h, bool(h.get("available")))
             for h in lib.get("home_assistant") or []]
    return [{"voice_id": v["voice_id"], "name": v["name"],
             "eligible": ok and (v.get("scope") or SCOPE_DJ) in DJ_SCOPES}
            for v, ok in rows]


def _next_dj(lineup: list[dict[str, Any]], current: str
             ) -> dict[str, Any] | None:
    """The first eligible voice AFTER `current` in the lineup, wrapping;
    the first eligible one when `current` is not on it. Never `current`
    itself: a selection whose recording changed is re-adopted by choosing
    it by name, never by a quick action."""
    order = [r["voice_id"] for r in lineup]
    # a current voice no shelf lists (the service default, or a voice
    # enrolled before the shelves) starts the walk at the top
    if current in order:
        i = order.index(current)
        ring = lineup[i + 1:] + lineup[:i]
    else:
        ring = lineup
    return next(({"voice_id": r["voice_id"], "name": r["name"]}
                 for r in ring if r["eligible"]), None)


def _dj_block(lib: dict[str, Any]) -> dict[str, Any]:
    """Who the DJ is, for a UI: the selection by name, whether it is in
    force (and what speaks when it is not), the voices that may be
    picked, and which one "Switch DJ" would move to."""
    force = dj_voice_in_force()
    lineup = _dj_lineup(lib)
    vid = force["selected_voice_id"]
    cur = lib.get("current") or {}
    name = (next((r["name"] for r in lineup if r["voice_id"] == vid), "")
            or (cur.get("name") if cur.get("voice_id") == vid else "")
            or (SERVICE_DEFAULT_NAME if vid in ("", SERVICE_DEFAULT_ID)
                else _display_name(vid, "")))
    if force.get("automatic"):
        # nobody has chosen: the tile names the voice actually speaking,
        # and says it was picked for them (automatic_dj_voice)
        sid = force["voice_id"]
        name = next((r["name"] for r in lineup if r["voice_id"] == sid),
                    "") or _display_name(sid, "")
    return {"voice_id": vid or None, "name": name,
            "automatic": bool(force.get("automatic")),
            "in_force": force["reason"] is None,
            "reason": force["reason"],
            "speaking_voice_id": force["voice_id"] or None,
            "speaking_name": (SERVICE_DEFAULT_NAME if force["reason"]
                              else name),
            "fallback_withheld": force["fallback_withheld"],
            "lineup": [{"voice_id": r["voice_id"], "name": r["name"]}
                       for r in lineup if r["eligible"]],
            "next": _next_dj(lineup, vid),
            "shifts": _shifts_block(lib)}


def switch_dj(fetch_fn=None, prove=None) -> dict[str, Any]:
    """"Switch DJ": select the next eligible voice in the lineup. A
    convenience over select(), never a second writer -- dj.persona.voice
    is written by select() alone, with every check it makes.

    The next voice is decided from the live shelf (library): allowed for
    the DJ, able to speak, custom voices included, wrapping at the end.
    Nothing is spoken and nothing reaches playback: the new voice is the
    one the next DJ break is rendered with.

    A current voice no shelf lists -- the service default, or a voice
    enrolled before the shelves (e.g. `legacy_voice_1`) -- is left exactly as it
    is until the owner acts; nothing migrates it on load. Pressing Switch
    DJ IS that act: the first eligible voice in the lineup is selected,
    and the rotation carries on from there. Such a voice cannot be chosen
    again afterwards unless it later exists on a shelf. When no other
    voice is eligible the selection stays and `changed` is False.

    `prove` (the API passes speak_check) must accept the next voice
    before it is selected; a voice that cannot speak is refused in words
    and the DJ stays as it was."""
    lib = library(fetch_fn)
    dj = lib.get("dj")
    if not dj:
        raise VoiceError("The DJ voice could not be read just now.")
    current = dj.get("voice_id") or ""
    nxt = dj.get("next")
    if nxt is None:
        return {"voice_id": current or None, "name": dj.get("name"),
                "previous_voice_id": current or None, "changed": False,
                "message": NO_OTHER_DJ_MESSAGE}
    if prove is not None:
        prove(nxt["voice_id"])
    out = select(nxt["voice_id"], fetch_fn)
    logger.info("DJ switched from %s to %s", current or "(none)",
                nxt["voice_id"])
    return {**out, "name": nxt["name"], "previous_voice_id": current or None}


# --- custom voices: an OPTIONAL capability -------------------------------
# Cloning is the one thing on this shelf that needs a component the base
# installation does not have. Everything below treats its absence as a
# normal state to be described, never as an error to be raised at a
# customer -- the Home Assistant and curated shelves stand either way.

def cloning_configured() -> bool:
    """Is a cloning engine configured at all? Configuration only; whether
    it ANSWERS is a separate question the inventory settles.

    The SAME resolver speech uses (providers.chatterbox_endpoint): the
    engine set under Settings > Optional services, else the environment.
    Reading only the environment told every installation whose engine
    lives in Settings -- every add-on -- that it had none: the Booth's Add
    voice stayed off and create/replace/delete answered "this installation
    has not got" one, while the same engine spoke every DJ break."""
    import providers
    url, _key = providers.chatterbox_endpoint()
    return bool(url)


def _require_engine() -> None:
    if not cloning_configured():
        raise VoiceEngineUnavailable(NO_ENGINE_MESSAGE)


def _require_enrollment() -> None:
    if not ENROLLMENT_QUALIFIED:
        raise VoiceEnrollmentUnavailable(ENROLLMENT_UNQUALIFIED_MESSAGE)


def cloning_status(inv: list[str] | None, *, unreachable: bool = False
                   ) -> dict[str, Any]:
    """What a Booth may say and offer about custom voices.

    `inv` is the live asset inventory (None when it could not be read).
    `available` means CREATE MAY BE OFFERED: an engine is configured and
    answering, and at least one slot is free. `reason` says why not, in
    terms a UI can branch on without re-deriving anything.
    """
    configured = cloning_configured()
    used = (None if inv is None
            else sum(1 for slot in CUSTOM_SLOTS if CUSTOM_ASSET[slot] in inv))
    out: dict[str, Any] = {
        "configured": configured,
        "available": False,
        "reason": "",
        "message": "",
        "processing": PROCESSING_NOTE,
        "slots_total": len(CUSTOM_SLOTS),
        "slots_used": used,
        "slots_free": None if used is None else len(CUSTOM_SLOTS) - used,
        "min_seconds": voice_sample_min_seconds(),
        "max_seconds": voice_sample_max_seconds(),
        "recommended": voice_sample_recommendation(),
    }
    if not ENROLLMENT_QUALIFIED:
        # First: with or without an engine, Create is not offered in this
        # release, and saying "add an engine" would promise that it is.
        out["reason"] = "not_qualified"
        out["message"] = ENROLLMENT_UNQUALIFIED_MESSAGE
        return out
    if not configured:
        out["reason"] = "unconfigured"
        out["message"] = CLONING_UNCONFIGURED
        return out
    if inv is None or unreachable:
        out["reason"] = "unreachable"
        out["message"] = CLONING_UNREACHABLE
        return out
    if out["slots_free"] == 0:
        out["reason"] = "slots_full"
        out["message"] = (
            f"This release holds {len(CUSTOM_SLOTS)} custom voices. Delete "
            "one to make room for another.")
        return out
    out["available"] = True
    return out


def voice_sample_min_seconds() -> int:
    import voice_sample_check
    return int(voice_sample_check.MIN_DURATION_S)


def voice_sample_max_seconds() -> int:
    import voice_sample_check
    return int(voice_sample_check.MAX_DURATION_S)


def voice_sample_recommendation() -> str:
    """The one place the recommended sample is described. The CHECK does
    not enforce this -- nothing can measure "one speaker, no music" -- so
    it lives here as guidance a UI shows, next to the bounds that are
    enforced."""
    import voice_sample_check
    return (f"About {voice_sample_check.RECOMMENDED_MIN_S}-"
            f"{voice_sample_check.RECOMMENDED_MAX_S} seconds of one person "
            "speaking clearly: no music, no other voices, little background "
            "noise.")


def first_free_slot(fetch_fn=None) -> str | None:
    """The first custom slot with no stored sample, or None when this
    release's three are all in use. Slot ids are STORAGE: they are chosen
    here and never asked for, shown, or accepted from a customer."""
    import tts_voices
    try:
        inv = set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable as exc:
        raise VoiceEngineUnavailable(CLONING_UNREACHABLE) from exc
    for slot in CUSTOM_SLOTS:
        if CUSTOM_ASSET[slot] not in inv:
            return slot
    return None


def _notes() -> dict[str, str]:
    import settings
    raw = settings.load()["dj"].get("voice_notes") or {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def set_note(voice_id: str, note: str) -> dict[str, Any]:
    """Attach the owner's own reminder to a voice. Any shelved voice --
    curated or custom -- may carry one; nothing else may, so a note can
    never be written for a voice that does not exist. Whitespace is
    collapsed; an empty note clears it."""
    import settings
    if not _is_shelved(voice_id):
        raise VoiceError(f"unknown voice: {voice_id}")
    note = " ".join(str(note or "").split())
    if len(note) > MAX_NOTE_LEN:
        raise VoiceError(f"note must be at most {MAX_NOTE_LEN} characters")
    settings.save_section("dj", {"voice_notes": {voice_id: note}})
    return {"voice_id": voice_id, "note": note}


def _names() -> dict[str, str]:
    """The owner's display names for built-ins, keyed by voice id."""
    import settings
    raw = settings.load()["dj"].get("voice_names") or {}
    return {k: v for k, v in raw.items() if isinstance(v, str) and v}


def _tuning() -> dict[str, dict[str, float]]:
    """Per-voice delivery, complete: every stored voice gets both fields."""
    import settings
    raw = settings.load()["dj"].get("voice_tuning") or {}
    out: dict[str, dict[str, float]] = {}
    for vid, tune in raw.items():
        if isinstance(tune, dict):
            out[vid] = {"speed": float(tune.get("speed", 1.0) or 1.0),
                        "pause": float(tune.get("pause", 1.0) or 1.0)}
    return out


def _tune_kwargs(speed: float, pause: float) -> dict[str, float]:
    """Provider kwargs for a delivery -- only what differs from 1.0."""
    kw: dict[str, float] = {}
    if speed != 1.0:
        kw["speed"] = speed
    if pause != 1.0:
        kw["silence_scale"] = pause
    return kw


def _check_tuning(speed: Any, pause: Any) -> tuple[float, float]:
    for name, val, (lo, hi) in (("speed", speed, SPEED_RANGE),
                                ("pause", pause, PAUSE_RANGE)):
        if (isinstance(val, bool) or not isinstance(val, (int, float))
                or not lo <= float(val) <= hi):
            raise VoiceError(f"{name} must be a number between {lo} and {hi}")
    return round(float(speed), 2), round(float(pause), 2)


def set_tuning(voice_id: str, speed: Any, pause: Any) -> dict[str, Any]:
    """Delivery for one built-in voice. Only Kokoro voices take it: a
    cloned voice's delivery is the recording's own. Stored whole, so a
    later read never has to guess a missing half."""
    import settings
    if not _is_kokoro(voice_id):
        raise VoiceError(f"only a built-in voice can be tuned: {voice_id}")
    speed, pause = _check_tuning(speed, pause)
    settings.save_section("dj", {"voice_tuning": {voice_id: {"speed": speed,
                                                              "pause": pause}}})
    return {"voice_id": voice_id, "speed": speed, "pause": pause}


def _custom_name(slot: str, cfg: dict[str, Any]) -> str:
    return cfg.get("name") or slot.replace("_", " ").title()


def _taken_names(except_id: str) -> set[str]:
    """Every display name on the shelf other than except_id's, casefolded.
    Names are what the HA select entity lists, so two rows may never share
    one -- picking by name would then be ambiguous."""
    import kokoro_voices
    names, slots = _names(), _slots()
    taken = {SERVICE_DEFAULT_NAME}
    known = set(CUSTOM_SLOTS)
    for v in kokoro_voices.shelf():
        known.add(v["voice_id"])
        if v["voice_id"] != except_id:
            taken.add(names.get(v["voice_id"]) or v["label"])
    for slot in CUSTOM_SLOTS:
        if slot != except_id:
            taken.add(_custom_name(slot, slots.get(slot) or {}))
    ha_voices, _st = _ha_shelf()
    for v in ha_voices:
        known.add(v["voice_id"])
        if v["voice_id"] != except_id:
            taken.add(names.get(v["voice_id"]) or v["name"])
    sel = _selected()
    vid = sel.get("id") or ""
    if vid and vid not in known:
        asset = str((sel.get("refs") or {}).get("chatterbox_voice") or "")
        taken.add(_display_name(vid, asset) if asset else SERVICE_DEFAULT_NAME)
    return {n.casefold() for n in taken}


def _slots() -> dict[str, Any]:
    import settings
    return settings.load()["dj"].get("voice_slots") or {}


def _selected() -> dict[str, Any]:
    import settings
    return settings.load()["dj"]["persona"].get("voice") or {}


# --- DJ shifts ---------------------------------------------------------------
# When dj.shifts.enabled, the DJ voice follows the station's three air shifts
# (dj_context.SHIFTS: Morning, Afternoon, Night, cut from the daypart
# boundaries, on the station clock). There is no second DJ setting and no
# second writer: at the first scheduling pass of a new shift occurrence the
# scheduled voice is chosen through select(), exactly as a person would.
#
# ONE persisted marker carries the whole state machine: dj.shifts.applied,
# the shift occurrence the selection was last settled for. The scheduler
# settles an occurrence once (reconcile_shift); an owner choosing a DJ --
# the picker or Switch DJ -- settles the CURRENT occurrence in the same
# write, so the scheduler leaves that choice alone until the next boundary.
# That is the temporary override, and it survives a restart because it is
# the marker. A shift change never speaks, never renders and never reaches
# Music Assistant: it only changes whose voice the NEXT DJ render uses.

#: Why a shift's assigned voice will not be chosen, beyond CHECK_REASONS.
SHIFT_UNAVAILABLE = "voice_unavailable"
#: A scheduled voice that merely could not speak at the boundary (Kokoro
#: or Home Assistant down for a moment) is asked again, at most this often,
#: instead of giving the whole shift to the previous DJ.
SHIFT_RETRY_S = 60.0
_SHIFT_RETRY_AT: dict[str, float] = {}


def _shifts() -> dict[str, Any]:
    import settings
    return settings.load()["dj"].get("shifts") or {}


def _shift_clock(now=None):
    """The station-local wall time (dj_context.shift_clock: the Home
    Assistant clock, never the container's or a browser's), or None."""
    import dj_context
    return dj_context.shift_clock(now)


def _manual_shift_key(now=None) -> str | None:
    """The shift occurrence an owner's choice of DJ settles, or None when
    shifts are off or the station clock is unknown."""
    try:
        if not _shifts().get("enabled"):
            return None
        local = _shift_clock(now)
        if local is None:
            return None
        import dj_context
        return dj_context.shift_instance(local)
    except Exception:  # noqa: BLE001 - never block a selection
        logger.debug("shift key unavailable", exc_info=True)
        return None


def shift_assignment_problem(assignment: dict[str, Any],
                             eligible: set[str] | None) -> str | None:
    """Why a shift's assigned voice may not be chosen right now, or None.
    An unshelved id is missing; then the DJ checks against the recording
    the assignment was made for (check_dj_voice); then, with a live lineup
    to ask, whether it can speak at all. An empty assignment is not a
    problem -- it is "no DJ scheduled", and the selection stays."""
    vid = str((assignment or {}).get("voice_id") or "")
    if not vid:
        return None
    if kind_of(vid) is None:
        return CHECK_MISSING
    reason = check_dj_voice(vid, (assignment or {}).get("identity") or None)
    if reason:
        return reason
    if eligible is not None and vid not in eligible:
        return SHIFT_UNAVAILABLE
    return None


def _eligible_ids(lib: dict[str, Any]) -> set[str]:
    return {r["voice_id"] for r in _dj_lineup(lib) if r["eligible"]}


def reconcile_shift(now=None, fetch_fn=None, *, force: bool = False
                    ) -> dict[str, Any]:
    """Settle the current shift occurrence, once. Called at the start of
    every DJ scheduling pass; a no-op read unless shifts are on, the
    station clock is known, and this occurrence has not been settled yet
    (a boundary was crossed, or shifts were just switched on).

    Settling chooses the shift's assigned voice through select() when it
    passes shift_assignment_problem; otherwise the current selection is
    kept (never a broader grant, never an unvetted recording). A voice that
    merely could not speak is asked again after SHIFT_RETRY_S rather than
    settled against. Writes dj.shifts.applied (and the selection) and
    nothing else; speaks nothing; touches no queue."""
    import dj_context
    import settings
    cfg = _shifts()
    if not cfg.get("enabled"):
        return {"outcome": "off"}
    local = _shift_clock(now)
    if local is None:
        return {"outcome": "clock_unknown"}
    key = dj_context.shift_instance(local)
    shift = dj_context.shift_of(local)
    if cfg.get("applied") == key and not force:
        return {"outcome": "settled", "shift": shift}
    assignment = cfg.get(shift) or {}
    vid = str(assignment.get("voice_id") or "")
    out: dict[str, Any] = {"outcome": "kept", "shift": shift,
                           "voice_id": vid or None, "problem": None}
    if vid:
        import time
        if not force and time.monotonic() < _SHIFT_RETRY_AT.get(key, 0.0):
            return {**out, "outcome": "retry_later"}
        problem = shift_assignment_problem(
            assignment, _eligible_ids(library(fetch_fn)))
        if problem is None:
            try:
                chosen = _select(vid, fetch_fn, key)
                out["outcome"] = "selected" if chosen.get("changed") else "already"
            except VoiceError as exc:
                problem = SHIFT_UNAVAILABLE
                logger.warning("DJ shift %s: %s could not be selected: %s",
                               shift, vid, str(exc)[:160])
        out["problem"] = problem
        if problem == SHIFT_UNAVAILABLE:
            _SHIFT_RETRY_AT[key] = time.monotonic() + SHIFT_RETRY_S
            logger.warning("DJ shift %s: %s cannot speak right now; the "
                           "current DJ stays and it is asked again", key, vid)
            return out
        _SHIFT_RETRY_AT.pop(key, None)
    if _shifts().get("applied") != key:
        settings.save_section("dj", {"shifts": {"applied": key}})
    logger.info("DJ shift %s settled: %s%s", key, out["outcome"],
                f" ({out['problem']})" if out["problem"] else "")
    return out


def set_shifts(enabled: Any = None, assignments: dict[str, Any] | None = None,
               fetch_fn=None, now=None, prove=None) -> dict[str, Any]:
    """Configure DJ shifts: switch them on or off, and/or assign a voice
    (a logical id, "" to unassign) to Morning, Afternoon or Night.

    Only a voice allowed for the DJ (dj or both) can be assigned -- an
    Announcement default grants nothing here either -- and a custom voice
    only with a recording, which the assignment is bound to (its identity)
    so that a later recording in the same slot is not adopted. Switching
    shifts ON, or changing the CURRENT shift's voice while they are on,
    settles the current shift now (reconcile_shift). Switching them OFF
    keeps whoever is the DJ and ends any override. Nothing is played.

    `prove` (the API passes speak_check) must accept every voice newly
    ASSIGNED before anything is saved: the owner learns now, in words,
    that a voice cannot speak -- not at a shift boundary in the night. The
    boundary itself only asks the cheap questions (shift_assignment_
    problem: still allowed, recording unchanged, provider available) and
    keeps the current DJ when one fails."""
    import dj_context
    import settings
    cfg = _shifts()
    patch: dict[str, Any] = {}
    for shift, vid in (assignments or {}).items():
        if shift not in dj_context.SHIFTS:
            raise VoiceError("A shift is Morning, Afternoon or Night.")
        vid = str(vid or "").strip()
        ident = ""
        if vid:
            if kind_of(vid) is None:
                raise VoiceError(f"unknown voice: {vid}")
            if scope_of(vid) not in DJ_SCOPES:
                raise VoiceNotAuthorized(
                    "That voice is allowed for announcements only, not as "
                    "the DJ.")
            if kind_of(vid) == CUSTOM_KIND:
                ident = identity_of(vid) or ""
                if not ident and not _slot_has_sample(vid, fetch_fn):
                    raise VoiceError("That voice has no recording stored yet.")
        patch[shift] = {"voice_id": vid, "identity": ident}
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise VoiceError("Automatic DJ shifts is on or off.")
        patch["enabled"] = enabled
        if not enabled:
            patch["applied"] = ""
    if prove is not None:
        for shift, a in patch.items():
            vid = a.get("voice_id") if isinstance(a, dict) else ""
            if vid and vid != str((cfg.get(shift) or {}).get("voice_id") or ""):
                prove(vid)
    now_on = patch.get("enabled", bool(cfg.get("enabled")))
    turning_on = now_on and not cfg.get("enabled")
    current_changed = False
    local = _shift_clock(now) if now_on else None
    if local is not None:
        cur = dj_context.shift_of(local)
        current_changed = cur in patch and patch[cur] != (cfg.get(cur) or {})
    settings.save_section("dj", {"shifts": patch})
    result = None
    if now_on and (turning_on or current_changed):
        result = reconcile_shift(now, fetch_fn, force=True)
    return {"shifts": _shifts_block(library(fetch_fn), now=now),
            "reconcile": result}


def _shifts_block(lib: dict[str, Any], now=None) -> dict[str, Any]:
    """Automatic DJ shifts as a UI shows them: whether on, the boundaries,
    the shift the station is in, each assignment by name with anything
    wrong with it, and whether the owner's own choice is holding the
    current shift (the override) until the next boundary. A pure read."""
    import dj_context
    cfg = _shifts()
    lineup = _dj_lineup(lib)
    names = {r["voice_id"]: r["name"] for r in lineup}
    eligible = {r["voice_id"] for r in lineup if r["eligible"]}
    local = _shift_clock(now)
    shift = dj_context.shift_of(local) if local is not None else None
    key = dj_context.shift_instance(local) if local is not None else None
    assignments = {}
    for s in dj_context.SHIFTS:
        a = cfg.get(s) or {}
        vid = str(a.get("voice_id") or "")
        assignments[s] = {
            "voice_id": vid or None,
            "name": (names.get(vid) or _display_name(vid, "")) if vid else None,
            "problem": shift_assignment_problem(a, eligible)}
    enabled = bool(cfg.get("enabled"))
    selected = str(_selected().get("id") or "")
    scheduled = assignments.get(shift, {}) if shift else {}
    override = bool(enabled and key and cfg.get("applied") == key
                    and scheduled.get("voice_id")
                    and scheduled.get("problem") is None
                    and selected != scheduled["voice_id"])
    return {"enabled": enabled,
            "clock_known": local is not None,
            "current_shift": shift,
            "next_shift": dj_context.next_shift(shift) if shift else None,
            "starts": {s: f"{h:02d}:00" for s, h in
                       dj_context.SHIFT_STARTS.items()},
            "assignments": assignments,
            "override": override,
            "pending": bool(enabled and key and cfg.get("applied") != key)}


def library(fetch_fn=None) -> dict[str, Any]:
    """The whole shelf, for a UI. Availability is live truth from the
    speech service; an unreadable inventory degrades honestly (ok=False)
    rather than offering voices nobody can prove exist."""
    import tts_voices
    out: dict[str, Any] = {"ok": True, "kokoro": [], "customs": [],
                           "home_assistant": [], "ha": None,
                           "current": None, "selected": None, "error": ""}
    try:
        inv = set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable as exc:
        inv = None
        if cloning_configured():
            # configured but not answering: the shelf is degraded. The
            # customer gets the settled sentence; the provider's own words
            # (which carry its URL) go to the log and stop there.
            out["ok"] = False
            out["error"] = CLONING_UNREACHABLE
            logger.warning("voice engine unreachable: %s", str(exc)[:200])
        else:
            # the base tier: no cloning engine at all is a normal state,
            # not a broken library -- the Home Assistant shelf still stands
            out["chatterbox"] = "unconfigured"
    notes, names, tuning, scopes = _notes(), _names(), _tuning(), _scopes()
    default = announcement_default()
    out["announcement_default"] = default
    out["preview"] = {"default_text": PREVIEW_PHRASE,
                      "max_chars": PREVIEW_MAX_CHARS}
    out["kokoro"] = _kokoro_shelf()
    for k in out["kokoro"]:
        k["builtin_name"] = k["name"]
        k["name"] = names.get(k["voice_id"]) or k["name"]
        k["note"] = notes.get(k["voice_id"], "")
        k["scope"] = scopes.get(k["voice_id"]) or SCOPE_DJ
        tune = tuning.get(k["voice_id"]) or {}
        k["speed"] = tune.get("speed", 1.0)
        k["pause"] = tune.get("pause", 1.0)
        k["announcement_default"] = k["voice_id"] == default["voice_id"]
    out["home_assistant"], out["ha"] = _ha_shelf()
    for h in out["home_assistant"]:
        h["builtin_name"] = h["name"]
        h["name"] = names.get(h["voice_id"]) or h["name"]
        h["note"] = notes.get(h["voice_id"], "")
        h["scope"] = scopes.get(h["voice_id"]) or SCOPE_DJ
        h["announcement_default"] = h["voice_id"] == default["voice_id"]
    slots = _slots()
    for slot in CUSTOM_SLOTS:
        cfg = slots.get(slot) or {}
        out["customs"].append({
            "voice_id": slot,
            "name": _custom_name(slot, cfg),
            "has_sample": (CUSTOM_ASSET[slot] in inv) if inv is not None
            else None,
            "note": notes.get(slot, ""),
            "scope": scopes.get(slot) or SCOPE_DJ,
            "announcement_default": slot == default["voice_id"],
            # whether the owner's acknowledgement is on record -- never
            # the fingerprint, the asset, or anything about the speaker
            "authorization": authorization_of(slot)})
    # The optional capability, described rather than assumed. A UI reads
    # this ONE block to decide whether to offer Create a Voice, to say why
    # not, and what to promise about where a recording is processed.
    out["cloning"] = cloning_status(None if inv is None else sorted(inv),
                                    unreachable=not out["ok"])
    # The acknowledgement a Create or Replace must show and send back.
    out["cloning"]["consent"] = {"text": CONSENT_TEXT,
                                 "version": CONSENT_VERSION}
    sel = _selected()
    vid = sel.get("id") or ""
    refs = sel.get("refs") or {}
    known = set(CUSTOM_SLOTS) | {k["voice_id"] for k in out["kokoro"]} \
        | {h["voice_id"] for h in out["home_assistant"]}
    if vid and _is_ha(vid) and vid not in known:
        # A selected Home Assistant voice while Home Assistant cannot be
        # listed: shown as itself, with the entity it names, unavailable.
        out["current"] = {"voice_id": vid,
                          "name": names.get(vid) or ha_entity_for(vid),
                          "provider": HA_PROVIDER,
                          "entity_id": ha_entity_for(vid),
                          "source": sel.get("source"), "available": False}
    elif vid and vid not in known:
        # An enrolled or legacy voice that predates this shelf, or the
        # speech service's own default after a deleted sample. It is REPORTED AS
        # ITSELF, never remapped onto some other shelf entry: a selector
        # that silently renames the customer's own voice is how a cloned
        # voice gets overwritten by a re-pick.
        asset = str(refs.get("chatterbox_voice") or "")
        out["current"] = {
            "voice_id": vid,
            "name": _display_name(vid, asset) if asset else SERVICE_DEFAULT_NAME,
            "provider": "chatterbox" if asset else "",
            "source": sel.get("source"),
            "available": (asset in inv) if (inv is not None and asset)
            else None,
        }
    out["selected"] = {"voice_id": vid or None, "source": sel.get("source"),
                       "provider": _selected_provider(refs)}
    try:
        out["dj"] = _dj_block(out)
    except Exception:  # noqa: BLE001 - a read for display only
        logger.debug("current DJ block failed", exc_info=True)
        out["dj"] = None
    return out


def resolve_asset(voice_id: str, fetch_fn=None) -> str:
    """voice_id -> the Chatterbox asset name, PROVEN against the live
    inventory. Raises VoiceError for unknown ids, empty custom slots, or
    an unreadable inventory (persisting or speaking an unverifiable voice
    would fail every break until someone noticed)."""
    import tts_voices
    if voice_id not in CUSTOM_SLOTS:
        raise VoiceError(f"unknown voice: {voice_id}")
    asset = CUSTOM_ASSET[voice_id]
    try:
        inv = tts_voices.inventory(fetch_fn)
    except tts_voices.VoiceCatalogUnavailable as exc:
        # An unconfigured engine and an unreachable one are different
        # answers to the customer, and neither is the provider's own
        # exception text.
        logger.warning("voice inventory unreadable: %s", str(exc)[:200])
        raise VoiceEngineUnavailable(
            NO_ENGINE_MESSAGE if not cloning_configured()
            else CLONING_UNREACHABLE) from exc
    if asset not in inv:
        raise VoiceError("That voice has no recording stored yet.")
    return asset


def select(voice_id: str, fetch_fn=None) -> dict[str, Any]:
    """Persist the selection. Writes dj.persona.voice ONLY -- personality
    and every other setting are structurally untouched.

    Which PROVIDER speaks is never chosen here and never submitted by a
    client: the registry owns that mapping, and the router reads it back
    off the refs this function writes.

    BOTH ref keys are always written, because `refs` MERGES on save. A
    selection that only named its own key would leave the previous
    provider's key standing, and the router -- which prefers a Kokoro ref
    whenever one is present -- would keep speaking in the voice the
    customer just switched away from.

    With automatic DJ shifts on, a selection also settles the current
    shift occurrence (dj.shifts.applied) in the SAME write: an owner's
    choice holds until the next shift boundary.
    """
    return _select(voice_id, fetch_fn, None)


def _select(voice_id: str, fetch_fn, shift_key: str | None
            ) -> dict[str, Any]:
    """select(), with the one thing only the shift scheduler says: WHICH
    occurrence this selection settles (reconcile_shift). The same body,
    the same checks, the same single write -- not a second writer."""
    import settings
    current = _selected()
    refs = dict(current.get("refs") or {})

    if voice_id and voice_id == (current.get("id") or "") and \
            check_dj_voice(voice_id, refs.get(IDENTITY_REF)) is None:
        # Re-picking what is already selected must be a no-op. Otherwise a
        # customer who opens the selector and presses the highlighted row
        # rewrites an enrolled voice's identity for no reason. A selection
        # that is NOT in force (its recording changed, its DJ scope was
        # withdrawn) is different: picking it again is the owner choosing
        # it as it is now, so it goes through every check below.
        return {"voice_id": current.get("id"), "source": current.get("source"),
                "chatterbox_voice": refs.get("chatterbox_voice", ""),
                "kokoro_voice": refs.get("kokoro_voice", ""),
                "ha_tts_entity": refs.get("ha_tts_entity", ""),
                "provider": _selected_provider(refs), "changed": False}

    if scope_of(voice_id) == SCOPE_ANNOUNCEMENT:
        # Granted for announcements ONLY: a voice the household recorded
        # for its own announcements is not thereby the station's DJ.
        raise VoiceError("That voice is allowed for announcements only, "
                         "not as the DJ voice.")

    # EVERY ref key is written on every selection: `refs` merges, and the
    # router prefers a Home Assistant ref, then a Kokoro ref, whenever one
    # is present -- a stale key would keep the old provider speaking.
    if _is_ha(voice_id):
        entity = ha_entity_for(voice_id)
        import providers
        if not providers.ha_configured():
            raise VoiceError("Home Assistant is not configured")
        try:
            import ha_tts
            ha_tts.entity_state(entity)      # proven to exist, like an asset
        except Exception as exc:  # noqa: BLE001 - the HA boundary
            raise VoiceError(f"Home Assistant voice {entity} unavailable: "
                             f"{str(exc)[:120]}") from exc
        new_refs = {"ha_tts_entity": entity, "kokoro_voice": "",
                    "chatterbox_voice": refs.get("chatterbox_voice", "")}
        identity = str(refs.get(IDENTITY_REF) or "")
        source = "builtin"
    elif _is_kokoro(voice_id):
        # The existing Chatterbox asset stays as the router's one bounded
        # fallback; it is not cleared just because a Kokoro voice is on.
        # Its fingerprint stays with it: a kept custom recording is held
        # to the DJ checks at every render (dj_voice_in_force).
        new_refs = {"kokoro_voice": voice_id, "ha_tts_entity": "",
                    "chatterbox_voice": refs.get("chatterbox_voice", "")}
        identity = str(refs.get(IDENTITY_REF) or "")
        source = "builtin"
    else:
        asset = resolve_asset(voice_id, fetch_fn)
        new_refs = {"chatterbox_voice": asset, "kokoro_voice": "",
                    "ha_tts_entity": ""}
        # WHICH recording was chosen: a later one in the same slot is
        # not this selection (check_dj_voice).
        identity = identity_of(voice_id) or ""
        source = "custom" if voice_id in CUSTOM_SLOTS else "builtin"
    if identity or refs.get(IDENTITY_REF):
        # written whenever there is a fingerprint to bind or a stale one
        # to clear -- refs MERGES, like every other key here
        new_refs[IDENTITY_REF] = identity

    patch: dict[str, Any] = {"persona": {"voice": {
        "id": voice_id, "source": source, "refs": new_refs}}}
    settles = shift_key or _manual_shift_key()
    if settles:
        patch["shifts"] = {"applied": settles}
    doc = settings.save_section("dj", patch)
    voice = doc["persona"]["voice"]
    saved = voice.get("refs") or {}
    return {"voice_id": voice["id"], "source": voice["source"],
            "chatterbox_voice": saved.get("chatterbox_voice", ""),
            "kokoro_voice": saved.get("kokoro_voice", ""),
            "ha_tts_entity": saved.get("ha_tts_entity", ""),
            "provider": _selected_provider(saved), "changed": True}


def rename(voice_id: str, name: str) -> dict[str, Any]:
    """A display name of the owner's choosing for any shelved voice. A
    custom slot keeps its name in voice_slots and must have one; a
    built-in's is an override in voice_names, and an empty name puts the
    curated name back. The name must be unique on the shelf. Names never
    reach a provider: selection and the persona hold the voice ID."""
    import kokoro_voices
    import settings
    if not _is_shelved(voice_id):
        raise VoiceError(f"unknown voice: {voice_id}")
    name = " ".join(str(name or "").split())
    if len(name) > MAX_NAME_LEN:
        raise VoiceError(f"name must be at most {MAX_NAME_LEN} characters")
    if voice_id in CUSTOM_SLOTS:
        if not name:
            raise VoiceError("a custom voice needs a name")
        shown = name
    elif _is_ha(voice_id):
        shown = name or next((v["name"] for v in _ha_shelf()[0]
                              if v["voice_id"] == voice_id), ha_entity_for(voice_id))
    else:
        shown = name or next(v["label"] for v in kokoro_voices.shelf()
                             if v["voice_id"] == voice_id)
    if shown.casefold() in _taken_names(voice_id):
        raise VoiceError(f"another voice is already called {shown!r}")
    if voice_id in CUSTOM_SLOTS:
        settings.save_section("dj", {"voice_slots": {voice_id: {"name": name}}})
    else:
        settings.save_section("dj", {"voice_names": {voice_id: name}})
    return {"voice_id": voice_id, "name": shown}


# --- enrollment: a recording becomes a voice ---------------------------------
# Every path that stores a sample -- create, replace, and the per-slot
# upload the automation service still uses -- runs the same steps in the
# same order, so a refusal at any step leaves nothing behind:
#
#   acknowledgement   the user confirmed they may create and use this voice
#   the audio         byte bounds, container magic, ffmpeg's measurements
#   the engine        the ONE copy of the sample lives in its store
#   confirmation      the registry can resolve the slot's asset afterwards
#   the record        identity, acknowledgement and scope, in one write
#
# A replacement first FORGETS which recording the slot held, then uploads.
# If anything after that fails, the slot is left with no identity, which a
# committed announcement reads as voice_missing and falls back from: it can
# never be the moment an announcement committed for the old recording is
# spoken in the new one.

#: When the engine accepted a recording and then did not list it.
NOT_CONFIRMED_MESSAGE = (
    "The ParlorWave Voice Engine did not confirm the recording, so the "
    "voice was not saved. Check that it is running, then try again.")
#: When a delete was accepted and the engine still lists the recording.
DELETE_NOT_CONFIRMED_MESSAGE = (
    "The ParlorWave Voice Engine still lists that recording, so the voice "
    "was not deleted. Check that it is running, then try again.")


class VoiceSampleRejected(VoiceError):
    """The recording itself cannot be used. `code` is stable per reason
    (sample_too_short, sample_mostly_silence, ...); the message says what
    to do about it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sniff(data: bytes) -> str:
    """The container by magic, never by filename. WAV/MP3/FLAC only.

    The three containers a browser's MediaRecorder produces (WebM, Ogg,
    MP4) are recognised only to say so: the Booth converts a recording to
    WAV on the owner's own device before sending it, so one of these
    arriving here is a client that skipped that step."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF
                              and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if (data[:4] == b"\x1aE\xdf\xa3" or data[:4] == b"OggS"
            or data[4:8] == b"ftyp"):
        raise VoiceSampleRejected(
            "sample_unsupported_format",
            "That recording is in a browser format (WebM, Ogg or MP4). "
            "Send it as a WAV, MP3 or FLAC file.")
    raise VoiceSampleRejected(
        "sample_unsupported_format",
        "That file is not a WAV, MP3 or FLAC recording.")


def _require_authorization(authorized: Any, consent_version: Any) -> None:
    """The user said yes -- to THIS acknowledgement. `authorized` must be
    the literal True, never a truthy string a client happened to send, and
    the version must name the text this release shows."""
    if authorized is not True or consent_version != CONSENT_VERSION:
        raise VoiceAuthorizationRequired(AUTHORIZATION_REQUIRED_MESSAGE)


def _check_source(source: Any) -> str:
    if source not in SAMPLE_SOURCES:
        raise VoiceError("A sample is either a recording or a file.")
    return source


def _authorization_record(source: str, now_fn=None) -> dict[str, str]:
    """What is kept about the acknowledgement: that it was given, to which
    text, when, and how the sample arrived. No speaker detail, no device,
    no address."""
    import datetime as _dt
    now = now_fn() if now_fn else _dt.datetime.now(_dt.timezone.utc)
    return {"status": AUTHORIZATION_GRANTED, "version": CONSENT_VERSION,
            "granted_at": now.astimezone(_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source}


def _empty_authorization() -> dict[str, str]:
    import settings
    return dict(settings.VOICE_AUTHORIZATION_EMPTY)


def authorization_of(voice_id: str) -> dict[str, Any]:
    """A custom voice's recorded acknowledgement, as a UI may show it:
    `granted` (with when, and how the sample arrived) or `unrecorded` --
    an empty slot, or a voice enrolled before the question was asked.
    Never the version string's meaning, never anything about the speaker."""
    auth = (_slots().get(voice_id) or {}).get("authorization") or {}
    if auth.get("status") == AUTHORIZATION_GRANTED:
        return {"status": AUTHORIZATION_GRANTED,
                "granted_at": auth.get("granted_at") or None,
                "source": auth.get("source") or None}
    return {"status": "unrecorded", "granted_at": None, "source": None}


def _validate_sample(data: bytes) -> tuple[str, dict[str, Any]]:
    """Byte bounds, container magic, then what ffmpeg can measure
    (voice_sample_check). Cheapest first; nothing reaches the engine until
    all three pass."""
    import voice_sample_check
    if not data or len(data) < MIN_SAMPLE_BYTES:
        raise VoiceSampleRejected(
            "sample_empty", "That recording is too small to be usable.")
    if len(data) > MAX_SAMPLE_BYTES:
        raise VoiceSampleRejected(
            "sample_too_large",
            f"That file is larger than {MAX_SAMPLE_BYTES // (1024 * 1024)} "
            "MB. A short MP3 or FLAC clip is well under it.")
    kind = _sniff(data)
    try:
        stats = voice_sample_check.check(data)
    except voice_sample_check.SampleRejected as exc:
        raise VoiceSampleRejected(f"sample_{exc.code}", str(exc)) from exc
    return kind, stats


def _store_sample(slot: str, data: bytes, kind: str, upload_fn=None,
                  delete_fn=None) -> None:
    """The engine's store holds the ONE copy of a sample. Nothing here
    writes audio anywhere else, and no byte of it is logged.

    For a slot that holds NO recording (create, or a first upload): a
    slot that already holds one is replaced by _replace_sample, which
    never deletes before the new recording is proven. Here a 409 can only
    mean the engine kept an asset under the slot's name that no voice
    owns (Chatterbox TTS API never overwrites: POST /voices answers 409
    for a held name); that orphan is deleted and the upload made once
    more."""
    import providers
    import tts_voices
    asset = CUSTOM_ASSET[slot]
    fn = upload_fn or providers.chatterbox_upload_voice
    try:
        try:
            fn(asset, data, f"{asset}.{kind}")
        except Exception as exc:  # noqa: BLE001 - the engine's HTTP answer
            # an HTTP 409 from the engine, recognised by its status alone:
            # this module neither builds nor fetches a URL
            if getattr(exc, "code", None) != 409:
                raise
            logger.info("voice engine already holds %s; replacing it", slot)
            (delete_fn or providers.chatterbox_delete_voice)(asset)
            fn(asset, data, f"{asset}.{kind}")
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - the engine boundary
        logger.warning("voice engine upload failed for %s: %s", slot,
                       str(exc)[:200])
        raise VoiceEngineFailed(ENGINE_FAILED_MESSAGE) from exc
    finally:
        tts_voices.reset_cache()


#: Where a replacement waits while it is proven, and where the recording
#: it replaces is set aside during the swap. Never a slot's own name, so
#: neither is ever listed, selected or spoken as a voice.
STAGED_SUFFIX = "_next"
ASIDE_SUFFIX = "_prev"
#: When the replacement never reached the voice it was meant for.
REPLACE_KEPT_MESSAGE = (
    "The ParlorWave Voice Engine could not take the new recording, so "
    "{name} keeps the recording it had. Check that the engine is running, "
    "then try again.")
#: When the engine stored the replacement and could not speak with it.
REPLACE_UNSPOKEN_MESSAGE = (
    "The ParlorWave Voice Engine stored the new recording but could not "
    "speak with it, so {name} keeps the recording it had. Try a clearer "
    "recording.")


def _fresh_inventory(fetch_fn=None) -> set[str]:
    import tts_voices
    tts_voices.reset_cache()
    try:
        return set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable as exc:
        raise VoiceEngineUnavailable(
            NO_ENGINE_MESSAGE if not cloning_configured()
            else CLONING_UNREACHABLE) from exc


def _put_back_set_aside(rename_fn=None, fetch_fn=None) -> None:
    """A replace that failed twice over -- the new recording would not
    move in AND the old one would not move back -- leaves the old
    recording under its set-aside name and the slot's own name empty.
    Before anything decides which slots are free or which voices have a
    recording, put any such recording back where it belongs. Best effort
    and never raises: an engine that cannot answer is reported by the
    step that follows, in that step's words."""
    import providers
    import tts_voices
    rename = rename_fn or providers.chatterbox_rename_voice
    try:
        inv = _fresh_inventory(fetch_fn)
    except VoiceError:
        return
    for slot in CUSTOM_SLOTS:
        asset = CUSTOM_ASSET[slot]
        if asset + ASIDE_SUFFIX in inv and asset not in inv:
            try:
                rename(asset + ASIDE_SUFFIX, asset)
                logger.warning("voice %s: put back the recording an earlier "
                               "replace had set aside", slot)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("voice %s: a set-aside recording could not "
                               "be put back: %s", slot, str(exc)[:160])
    tts_voices.reset_cache()


def _keep_outgoing(slot: str) -> None:
    """The Voice Library's copy of the slot's recording, before anything
    replaces or removes it. Raises (VoiceError) when it cannot be made."""
    import voice_collection
    voice_collection.keep_slot(slot)


def _replace_sample(slot: str, data: bytes, kind: str, *, upload_fn=None,
                    delete_fn=None, rename_fn=None, synth_fn=None,
                    fetch_fn=None, keep_fn=None) -> None:
    """Put a new recording under a slot that already holds one WITHOUT
    ever leaving the owner with neither.

    The engine never overwrites (POST /voices answers 409 for a held
    name), and a slot's asset name is fixed. Deleting the old recording
    and uploading the new one -- the previous way -- lost the only working
    recording whenever the upload then failed. Instead:

      1. the new recording is uploaded under the STAGED name, confirmed in
         a fresh inventory, and must SPEAK one short phrase;
      2. only then is the old recording renamed ASIDE and the staged one
         renamed into the slot's name -- the engine's own rename moves the
         file and its entry together;
      3. the set-aside recording is deleted last.

    Any failure in 1 removes the staged upload and leaves the slot exactly
    as it was. A failure in 2 renames the old recording back. The slot's
    fingerprint is cleared for the swap itself (fail-closed: a crash in
    that instant shows the voice as missing, never as the old grant over a
    new recording) and restored if the swap is undone. A replace that an
    earlier attempt left half-done -- a set-aside recording and no slot
    recording -- is put back before anything else, so the old voice is
    never stranded under the set-aside name.

    Before any of it, the recording the slot holds is KEPT in the Voice
    Library (keep_fn, default _keep_outgoing): a copy proven against the
    engine's listing. If it cannot be kept, nothing is uploaded and the
    slot keeps what it had -- so a replace never costs the old recording."""
    import providers
    import settings
    import tts_voices
    upload = upload_fn or providers.chatterbox_upload_voice
    delete = delete_fn or providers.chatterbox_delete_voice
    rename = rename_fn or providers.chatterbox_rename_voice
    synth = synth_fn or providers.chatterbox_synthesize
    asset = CUSTOM_ASSET[slot]
    staged, aside = asset + STAGED_SUFFIX, asset + ASIDE_SUFFIX
    name = _custom_name(slot, _slots().get(slot) or {}) or "The voice"
    kept = VoiceEngineFailed(REPLACE_KEPT_MESSAGE.format(name=name))

    def quietly(what, fn, *args) -> bool:
        try:
            fn(*args)
            return True
        except Exception as exc:  # noqa: BLE001 - already failing or tidying
            logger.warning("voice replace for %s: %s failed: %s", slot, what,
                           str(exc)[:160])
            return False

    try:
        inv = _fresh_inventory(fetch_fn)
        if aside in inv and asset not in inv:
            # an earlier swap stopped half-way: the old recording first
            rename(aside, asset)
            logger.warning("voice replace for %s: put back a recording an "
                           "earlier replace had set aside", slot)
        elif aside in inv:
            delete(aside)
        if staged in inv:
            delete(staged)
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - the engine boundary
        logger.warning("voice replace for %s could not tidy the engine: %s",
                       slot, str(exc)[:160])
        raise kept from exc
    # an injected engine (a test, a harness) brings its own keep, exactly
    # as it brings its own engine checks (upload_sample: _require_engine)
    keep = keep_fn or (_keep_outgoing if upload_fn is None else None)
    if keep:
        keep(slot)
    try:
        try:
            upload(staged, data, f"{staged}.{kind}")
        except Exception as exc:  # noqa: BLE001 - the engine boundary
            logger.warning("voice replace for %s: upload failed: %s", slot,
                           str(exc)[:160])
            raise kept from exc
        if staged not in _fresh_inventory(fetch_fn):
            logger.warning("voice replace for %s: the engine did not list "
                           "the new recording", slot)
            raise kept
        try:
            audio = synth(SPEAK_CHECK_TEXT, staged)
        except Exception as exc:  # noqa: BLE001 - the engine boundary
            logger.warning("voice replace for %s: the new recording did not "
                           "speak: %s", slot, str(exc)[:160])
            raise VoiceEngineFailed(
                REPLACE_UNSPOKEN_MESSAGE.format(name=name)) from exc
        if not audio or len(audio) < 128:
            raise VoiceEngineFailed(REPLACE_UNSPOKEN_MESSAGE.format(name=name))
    except VoiceError:
        quietly("removing the staged recording", delete, staged)
        tts_voices.reset_cache()
        raise
    old_identity = identity_of(slot) or ""
    _forget_identity(slot)
    try:
        rename(asset, aside)
    except Exception as exc:  # noqa: BLE001 - nothing has moved yet
        logger.warning("voice replace for %s: could not set the old "
                       "recording aside: %s", slot, str(exc)[:160])
        if old_identity:
            settings.save_section("dj", {"voice_slots": {slot: {
                "identity": old_identity}}})
        quietly("removing the staged recording", delete, staged)
        tts_voices.reset_cache()
        raise kept from exc
    try:
        rename(staged, asset)
    except Exception as exc:  # noqa: BLE001 - undo the half-swap
        logger.warning("voice replace for %s: could not move the new "
                       "recording in: %s", slot, str(exc)[:160])
        if quietly("putting the old recording back", rename, aside, asset):
            if old_identity:
                settings.save_section("dj", {"voice_slots": {slot: {
                    "identity": old_identity}}})
            quietly("removing the staged recording", delete, staged)
        # else: the old recording waits under the set-aside name; the slot
        # shows as missing (no fingerprint) and the next replace puts the
        # recording back before anything else
        tts_voices.reset_cache()
        raise kept from exc
    quietly("removing the old recording", delete, aside)
    tts_voices.reset_cache()


def _confirm_stored(slot: str, fetch_fn=None) -> None:
    """Success is what the registry can resolve, not what the engine said:
    a FRESH inventory must list the slot's asset."""
    import tts_voices
    tts_voices.reset_cache()
    try:
        resolve_asset(slot, fetch_fn)
    except VoiceError as exc:
        logger.warning("voice engine did not confirm the sample for %s: %s",
                       slot, type(exc).__name__)
        raise VoiceEngineFailed(NOT_CONFIRMED_MESSAGE) from exc


def _forget_identity(slot: str) -> None:
    """Before a slot's recording is replaced: it holds nobody we can vouch
    for until the new one is confirmed. Fail-closed by design."""
    import settings
    if identity_of(slot):
        settings.save_section("dj", {"voice_slots": {slot: {"identity": ""}}})


def _has_sample_strict(slot: str, fetch_fn=None) -> bool:
    """Does the slot hold a recording? An unreadable inventory is an
    unavailable engine, never a "no"."""
    import tts_voices
    try:
        return CUSTOM_ASSET[slot] in set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable as exc:
        raise VoiceEngineUnavailable(
            NO_ENGINE_MESSAGE if not cloning_configured()
            else CLONING_UNREACHABLE) from exc


def upload_sample(slot: str, data: bytes, upload_fn=None, *,
                  authorized: Any = False, consent_version: Any = "",
                  source: Any = "file", fetch_fn=None, now_fn=None,
                  delete_fn=None, rename_fn=None, synth_fn=None
                  ) -> dict[str, Any]:
    """Store a sample for one custom slot via the speech service's own
    store -- the per-slot primitive the automation service uses. Replaces
    any previous sample for the slot (same fixed asset name). The slot's
    display name and the current selection are untouched.

    Refused without the owner's acknowledgement (VoiceAuthorizationRequired)
    before the audio is even looked at. An engine that is not configured
    is a VoiceEngineUnavailable BEFORE any of that -- the customer should
    not be told their recording is too quiet when the real answer is that
    there is nothing to clone with. A slot that held no recording keeps no
    grant from whoever held it last: its scope starts at "dj"."""
    _require_enrollment()
    if slot not in CUSTOM_SLOTS:
        raise VoiceError(f"unknown custom slot: {slot}")
    if upload_fn is None:
        _require_engine()
    _require_authorization(authorized, consent_version)
    source = _check_source(source)
    kind, stats = _validate_sample(data)
    return _put_sample(slot, data, kind, stats=stats, authorized=authorized,
                       consent_version=consent_version, source=source,
                       upload_fn=upload_fn, fetch_fn=fetch_fn, now_fn=now_fn,
                       delete_fn=delete_fn, rename_fn=rename_fn,
                       synth_fn=synth_fn)


def _put_sample(slot: str, data: bytes, kind: str, *,
                stats: dict[str, Any] | None = None, authorized: Any = False,
                consent_version: Any = "", source: Any = "file",
                keep_fn=None, upload_fn=None, fetch_fn=None, now_fn=None,
                delete_fn=None, rename_fn=None, synth_fn=None
                ) -> dict[str, Any]:
    """upload_sample after its audio checks: also the Voice Library's way
    into a slot (voice_collection.assign), whose recordings were checked
    when they were imported or were already spoken by the engine. The
    gates are asked again, so no caller reaches the engine without them."""
    import settings
    _require_enrollment()
    if slot not in CUSTOM_SLOTS:
        raise VoiceError(f"unknown custom slot: {slot}")
    if upload_fn is None:
        _require_engine()
    _require_authorization(authorized, consent_version)
    source = _check_source(source)
    stats = stats or {}
    _put_back_set_aside(rename_fn, fetch_fn)
    had_sample = _slot_has_sample(slot, fetch_fn)
    if had_sample:
        # a recording is there: the same swap a Booth replace makes, so a
        # failed upload never costs the slot the recording it had
        _replace_sample(slot, data, kind, upload_fn=upload_fn,
                        delete_fn=delete_fn, rename_fn=rename_fn,
                        synth_fn=synth_fn, fetch_fn=fetch_fn, keep_fn=keep_fn)
    else:
        _forget_identity(slot)
        _store_sample(slot, data, kind, upload_fn, delete_fn)
    _confirm_stored(slot, fetch_fn)
    patch: dict[str, Any] = {"voice_slots": {slot: {
        "identity": sample_identity(data),
        "authorization": _authorization_record(source, now_fn)}}}
    if not had_sample:
        patch["voice_scopes"] = {slot: SCOPE_DJ}
    else:
        if _selected().get("id") == slot:
            _rebind_dj(patch, sample_identity(data))
        _rebind_shifts(patch, slot, sample_identity(data))
    settings.save_section("dj", patch)
    logger.info("voice sample stored for %s (%d bytes, %s, %s)", slot,
                len(data), kind, stats.get("analysis"))
    return {"voice_id": slot, "asset": CUSTOM_ASSET[slot], "bytes": len(data),
            "format": kind, "seconds": stats.get("duration_s"),
            "warnings": stats.get("warnings") or []}


def create_voice(name: str, data: bytes, note: str = "", *,
                 authorized: Any = False, consent_version: Any = "",
                 scope: Any = SCOPE_DJ, source: Any = "file",
                 announcement_default: bool = False,
                 upload_fn=None, fetch_fn=None, delete_fn=None,
                 now_fn=None, rename_fn=None, synth_fn=None
                 ) -> dict[str, Any]:
    """Create a custom voice from a recording: the WHOLE enrollment, as
    one refusable step.

    The customer names a voice, acknowledges they may use it, and says
    what it is for; this picks the storage. Which of the three slots a
    voice lands in is an implementation detail of this release, so the
    caller neither chooses it nor sees it -- when dynamic voice storage
    replaces the slots, only this function changes.

    Order is chosen so a refusal leaves NOTHING behind: engine, the
    acknowledgement, the request's own fields, a free slot, the name, the
    audio, and only then the upload. After the upload the voice must be
    CONFIRMED in a fresh inventory, and then its whole record -- name,
    identity, acknowledgement, scope, note -- is written at once. Any
    failure after the upload removes the sample again: a voice either
    exists completely or not at all.

    `scope` is what the user chose, written explicitly: a slot's previous
    occupant's grant is never inherited. `announcement_default` also makes
    it the default announcement voice, which needs announcement scope --
    choosing a default never grants one.

    `synth_fn` is accepted so ONE engine seam serves create, replace and
    upload alike; a create fills an empty slot and never needs it.
    `rename_fn` puts back a recording a failed replace left set aside
    (_put_back_set_aside) before a free slot is chosen."""
    import settings
    del synth_fn
    _require_enrollment()
    _require_engine()
    _require_authorization(authorized, consent_version)
    source = _check_source(source)
    if scope not in settings.VOICE_SCOPES:
        raise VoiceError("Choose what the voice is for: the DJ, "
                         "announcements, or both.")
    if announcement_default and scope not in ANNOUNCEMENT_SCOPES:
        raise VoiceNotAuthorized(
            "Only a voice allowed for announcements can be the default "
            "announcement voice.")
    name = " ".join(str(name or "").split())
    if not name:
        raise VoiceError("Give the voice a name.")
    if len(name) > MAX_NAME_LEN:
        raise VoiceError(f"A name can be at most {MAX_NAME_LEN} characters.")
    note = " ".join(str(note or "").split())[:MAX_NOTE_LEN]
    _put_back_set_aside(rename_fn, fetch_fn)
    slot = first_free_slot(fetch_fn)
    if slot is None:
        raise VoiceSlotsFull(
            f"This release holds {len(CUSTOM_SLOTS)} custom voices. Delete "
            "one to make room for another.")
    if name.casefold() in _taken_names(slot):
        # a sentence, not a repr: !r would put Python's quoting in front
        # of somebody reading a form
        raise VoiceNameTaken(f'Another voice is already called "{name}".')
    kind, stats = _validate_sample(data)
    _store_sample(slot, data, kind, upload_fn, delete_fn)
    try:
        _confirm_stored(slot, fetch_fn)
        settings.save_section("dj", {
            "voice_slots": {slot: {
                "name": name, "identity": sample_identity(data),
                "authorization": _authorization_record(source, now_fn)}},
            "voice_scopes": {slot: scope},
            "voice_notes": {slot: note}})
        default_now = str(settings.announcements_config()
                          .get("default_voice_id") or "")
        if announcement_default:
            settings.save_section("station", {"announcements": {
                "default_voice_id": slot}})
        elif default_now == slot:
            # a default left pointing at an EMPTY slot (its recording
            # vanished outside Parlorwave) must not quietly adopt whoever
            # is enrolled into it next
            settings.save_section("station", {"announcements": {
                "default_voice_id": ""}})
    except Exception as exc:
        _roll_back(slot, delete_fn)
        if isinstance(exc, VoiceError):
            raise
        logger.warning("voice enrollment failed after upload for %s: %s",
                       slot, type(exc).__name__)
        raise VoiceEngineFailed(ENGINE_FAILED_MESSAGE) from exc
    logger.info("custom voice created in %s (%s, %s)", slot, scope, source)
    return {"voice_id": slot, "name": name, "note": note, "scope": scope,
            "announcement_default": bool(announcement_default),
            "authorization": authorization_of(slot),
            "seconds": stats.get("duration_s"),
            "warnings": stats.get("warnings") or [],
            "slots_free": sum(1 for s in CUSTOM_SLOTS
                              if s != slot and not _slot_has_sample(s, fetch_fn))}


def replace_voice(voice_id: str, data: bytes, *, authorized: Any = False,
                  consent_version: Any = "", scope: Any = None,
                  source: Any = "file", upload_fn=None, fetch_fn=None,
                  now_fn=None, delete_fn=None, rename_fn=None,
                  synth_fn=None, keep_fn=None) -> dict[str, Any]:
    """Put a new recording under an existing custom voice, on purpose.

    The name, the note and the DJ selection stay; the IDENTITY changes,
    because it is a different recording -- so an announcement committed
    against the old one fails its render-time check and falls back rather
    than speaking in the new one. The acknowledgement is asked again (it
    is a new recording), and `scope`, when given, is re-chosen with it;
    omitted, the voice keeps what it was allowed. Only a voice that holds
    a recording can be replaced; an empty one is created instead.

    If this voice IS the DJ, the DJ follows the new recording once it is
    confirmed (_rebind_dj) -- replacing it was the owner's deliberate act.
    A replacement that fails at any point leaves the voice with the
    recording, fingerprint and acknowledgement it had (_replace_sample):
    the owner's only working recording is never the price of a bad
    upload."""
    import settings
    _require_enrollment()
    if voice_id not in CUSTOM_SLOTS:
        raise VoiceError("Only a voice made from a recording can be "
                         "replaced.")
    if upload_fn is None:
        _require_engine()
    _require_authorization(authorized, consent_version)
    source = _check_source(source)
    if scope is not None:
        _check_scope(voice_id, scope)
    _put_back_set_aside(rename_fn, fetch_fn)
    if not _has_sample_strict(voice_id, fetch_fn):
        raise VoiceError("That voice has no recording to replace. Create a "
                         "new voice instead.")
    kind, stats = _validate_sample(data)
    _replace_sample(voice_id, data, kind, upload_fn=upload_fn,
                    delete_fn=delete_fn, rename_fn=rename_fn,
                    synth_fn=synth_fn, fetch_fn=fetch_fn, keep_fn=keep_fn)
    _confirm_stored(voice_id, fetch_fn)
    patch: dict[str, Any] = {"voice_slots": {voice_id: {
        "identity": sample_identity(data),
        "authorization": _authorization_record(source, now_fn)}}}
    if scope is not None:
        patch["voice_scopes"] = {voice_id: scope}
    if _selected().get("id") == voice_id:
        _rebind_dj(patch, sample_identity(data))
    _rebind_shifts(patch, voice_id, sample_identity(data))
    settings.save_section("dj", patch)
    logger.info("custom voice %s replaced (%s)", voice_id, source)
    return {"voice_id": voice_id,
            "name": _custom_name(voice_id, _slots().get(voice_id) or {}),
            "scope": scope_of(voice_id), "replaced": True,
            "authorization": authorization_of(voice_id),
            "seconds": stats.get("duration_s"),
            "warnings": stats.get("warnings") or []}


def _rebind_dj(patch: dict[str, Any], identity: str) -> None:
    """The owner re-recorded the voice that IS the DJ, on purpose and with
    a fresh acknowledgement: the DJ stays, and from this write names the
    new recording. In the SAME write as the new fingerprint, so there is
    no moment the two disagree. Only the selected slot: a custom voice
    merely kept as the Chatterbox fallback is not re-adopted -- it is
    withheld until the owner chooses it again (dj_voice_in_force)."""
    patch["persona"] = {"voice": {"refs": {IDENTITY_REF: identity}}}


def _rebind_shifts(patch: dict[str, Any], slot: str, identity: str) -> None:
    """The same deliberate re-recording, for every DJ shift the voice is
    assigned to: the assignment follows the new recording. A recording
    that arrives any other way (a new voice enrolled into a slot a shift
    still names) is never adopted -- its identity will not match."""
    for shift, a in _shifts().items():
        if isinstance(a, dict) and a.get("voice_id") == slot:
            patch.setdefault("shifts", {})[shift] = {"voice_id": slot,
                                                     "identity": identity}


def _clear_slot(slot: str) -> bool:
    """Everything Parlorwave keeps about a slot's voice, gone: name,
    identity, acknowledgement, scope, note -- and the default announcement
    voice, if it was this one. Returns whether the default was cleared."""
    import settings
    patch: dict[str, Any] = {
        "voice_slots": {slot: {"name": "", "identity": "",
                               "authorization": _empty_authorization()}},
        "voice_scopes": {slot: SCOPE_DJ},
        "voice_notes": {slot: ""}}
    # a deleted voice is no longer anybody's shift, like the default
    for shift, a in _shifts().items():
        if isinstance(a, dict) and a.get("voice_id") == slot:
            patch.setdefault("shifts", {})[shift] = {"voice_id": "",
                                                     "identity": ""}
    settings.save_section("dj", patch)
    if str(settings.announcements_config().get("default_voice_id") or "") == slot:
        settings.save_section("station", {"announcements": {
            "default_voice_id": ""}})
        return True
    return False


def _roll_back(slot: str, delete_fn=None) -> None:
    """Undo a half-made voice after its upload succeeded. Best effort at
    the engine (logged, never raised over the original failure); the
    settings record is cleared regardless, so a sample the engine kept
    shows up as an unnamed, unauthorized recording -- never as the voice
    that failed."""
    import providers
    import tts_voices
    try:
        (delete_fn or providers.chatterbox_delete_voice)(CUSTOM_ASSET[slot])
    except Exception as exc:  # noqa: BLE001 - already failing
        logger.warning("voice rollback could not delete %s: %s", slot,
                       type(exc).__name__)
    tts_voices.reset_cache()
    try:
        _clear_slot(slot)
    except Exception:  # noqa: BLE001 - already failing
        logger.warning("voice rollback could not clear %s", slot,
                       exc_info=True)


def _slot_has_sample(slot: str, fetch_fn=None) -> bool:
    import tts_voices
    try:
        return CUSTOM_ASSET[slot] in set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable:
        return False


def delete_sample(slot: str, delete_fn=None, fetch_fn=None, keep_fn=None
                  ) -> dict[str, Any]:
    """Remove a custom voice: its recording from the engine's store, and
    everything Parlorwave keeps about it (_clear_slot). If the slot is the
    SELECTED DJ voice, selection falls back to the speech service's own
    default -- a break must never try to speak with a deleted sample, and
    it must never be handed a curated voice the customer did not pick.

    The engine is asked again afterwards when it can be: a recording it
    still lists was not deleted, and nothing is cleared. An announcement
    committed for the voice then finds no identity at render and falls
    back (voice_missing)."""
    import providers
    import settings
    import tts_voices
    if slot not in CUSTOM_SLOTS:
        raise VoiceError(f"unknown custom slot: {slot}")
    if delete_fn is None:
        _require_engine()
    # the recording stays in the Voice Library: removing a voice from its
    # slot is not deleting the recording
    keep = keep_fn or (_keep_outgoing if delete_fn is None else None)
    if keep:
        keep(slot)
    try:
        (delete_fn or providers.chatterbox_delete_voice)(CUSTOM_ASSET[slot])
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - the engine boundary
        logger.warning("voice engine delete failed for %s: %s", slot,
                       str(exc)[:200])
        raise VoiceEngineFailed(ENGINE_FAILED_MESSAGE) from exc
    tts_voices.reset_cache()
    try:
        still_listed = CUSTOM_ASSET[slot] in set(tts_voices.inventory(fetch_fn))
    except tts_voices.VoiceCatalogUnavailable:
        still_listed = False        # the delete was accepted; cannot re-ask
    if still_listed:
        logger.warning("voice engine still lists %s after a delete", slot)
        raise VoiceEngineFailed(DELETE_NOT_CONFIRMED_MESSAGE)
    default_cleared = _clear_slot(slot)
    import voice_collection
    voice_collection.forget_slot(slot)
    out = {"voice_id": slot, "deleted": True, "selection_reset": False,
           "announcement_default_cleared": default_cleared}
    if _selected().get("id") == slot:
        # BOTH ref keys cleared explicitly: `refs` merges, and a deleted
        # custom sample must land on the service default, never on
        # whichever Kokoro voice happened to be selected earlier.
        settings.save_section("dj", {"persona": {"voice": {
            "id": SERVICE_DEFAULT_ID, "source": "builtin",
            "refs": {"chatterbox_voice": "", "kokoro_voice": "",
                     "ha_tts_entity": "", IDENTITY_REF: ""}}}})
        out["selection_reset"] = True
        out["selected"] = SERVICE_DEFAULT_ID
    logger.info("custom voice deleted from %s", slot)
    return out


def preview_text(text: Any) -> str:
    """The line a preview speaks: the owner's own, cleaned and bounded, or
    the fixed phrase when they typed nothing. Control characters become
    spaces and runs of space collapse; nothing else about it changes."""
    if text is None:
        return PREVIEW_PHRASE
    if not isinstance(text, str):
        raise VoiceError("A preview line must be text.")
    cleaned = " ".join("".join(ch if ch.isprintable() else " "
                               for ch in text).split())
    if not cleaned:
        return PREVIEW_PHRASE
    if len(cleaned) > PREVIEW_MAX_CHARS:
        raise VoiceError(f"A preview line can be at most {PREVIEW_MAX_CHARS} "
                         "characters.")
    return cleaned


def preview(voice_id: str, synth_fn=None, fetch_fn=None, *,
            speed: Any = None, pause: Any = None, text: Any = None) -> bytes:
    """One short line in the given voice: audio bytes for the UI to play
    locally. No Music Assistant, no break, no announcement, no history,
    no cadence -- this never leaves the management surface.

    `text` is a line the owner typed (preview_text bounds it); without one
    the fixed phrase is spoken. A built-in voice previews with its stored
    delivery, or with the `speed`/`pause` given here so the owner can hear
    a setting before saving it. Neither applies to a cloned voice, which
    is spoken through tts_router.synthesize_voice -- the SAME call an
    announcement naming that voice makes, so what the owner hears is what
    a calendar event would get."""
    import providers
    phrase = preview_text(text)
    if _is_ha(voice_id):
        fn = synth_fn or providers.ha_tts_synthesize
        return fn(phrase, ha_entity_for(voice_id), timeout=PREVIEW_TIMEOUT_S)
    if _is_kokoro(voice_id):
        import kokoro_voices
        fn = synth_fn or providers.kokoro_synthesize
        stored = _tuning().get(voice_id) or {"speed": 1.0, "pause": 1.0}
        s, p = _check_tuning(stored["speed"] if speed is None else speed,
                             stored["pause"] if pause is None else pause)
        return fn(phrase, kokoro_voices.kokoro_id_for(voice_id),
                  timeout=PREVIEW_TIMEOUT_S, **_tune_kwargs(s, p))
    if voice_id not in CUSTOM_SLOTS:
        raise VoiceError(f"unknown voice: {voice_id}")
    import tts_router
    fn = synth_fn or providers.chatterbox_synthesize
    try:
        return tts_router.synthesize_voice(
            phrase, voice_id, fetch_fn=fetch_fn,
            chatterbox_fn=lambda t, asset: fn(t, asset,
                                               timeout=PREVIEW_TIMEOUT_S))
    except tts_router.VoiceUnavailable as exc:
        if exc.reason == tts_router.UNAVAILABLE_ENGINE:
            raise VoiceEngineUnavailable(
                NO_ENGINE_MESSAGE if not cloning_configured()
                else CLONING_UNREACHABLE) from exc
        if exc.reason == tts_router.UNAVAILABLE_NO_SAMPLE:
            raise VoiceError("That voice has no recording stored yet.") from exc
        logger.warning("voice engine preview failed for %s: %s", voice_id,
                       exc.reason)
        raise VoiceEngineFailed(ENGINE_FAILED_MESSAGE) from exc
