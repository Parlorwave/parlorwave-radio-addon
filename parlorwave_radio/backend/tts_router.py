"""Which speech provider says this break, and what happens when it can't.

This is a ROUTER, not a plugin framework. There are exactly three providers
and they are named here in full; adding one is a deliberate edit, not a
registration call.

    a Home Assistant TTS voice    -> Home Assistant, render-only (ha_tts);
                                     no fallback: a failure is reported as
                                     exactly that
    a curated Kokoro built-in     -> Kokoro sidecar
    everything else               -> Chatterbox, byte-for-byte as before

"Everything else" is any existing installation: the legacy
single-voice selection, the cloned custom slots, and a fresh install
whose refs are empty and which therefore lets the speech service pick its
own default sample. None of those resolve differently because this module
exists -- when no Kokoro voice is selected, or the kill switch is off,
the call made is exactly the call render_break made before.

FALLBACK is one-way and one-shot. If Kokoro is selected but cannot speak,
we try Chatterbox ONCE, and only when the settings already carry a proven
Chatterbox asset to fall back to. Chatterbox failing is the end of it:
there is no path back to Kokoro, so the two providers can never ping-pong.
A break that cannot be spoken simply fails, which the break lifecycle
already turns into a skipped element with the music untouched -- that is
the existing safety property and this module does not weaken it.

Deliberately absent: retries. The engine has no TTS retry today and a
90-second cooldown already stops a hard-down provider being hammered.
A fallback is not a retry -- it is one attempt at a different provider.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import kokoro_voices
import providers

logger = logging.getLogger("musicbrain")

#: settings key under dj.persona.voice.refs naming a curated built-in.
#: `refs` is already the schema-extensible map, so this needs no schema
#: change and leaves refs.chatterbox_voice exactly as it was.
KOKORO_REF = "kokoro_voice"
CHATTERBOX_REF = "chatterbox_voice"
#: settings key naming the Home Assistant TTS entity (tts.piper ...) a
#: selected Home Assistant voice speaks through
HA_REF = "ha_tts_entity"

PROVIDER_KOKORO = "kokoro"
PROVIDER_CHATTERBOX = "chatterbox"
PROVIDER_HOME_ASSISTANT = "home_assistant"


def _refs() -> dict[str, Any]:
    """dj.persona.voice.refs, or {} if settings cannot be read. A settings
    failure must never block a break: it degrades to the Chatterbox path
    the installation already had."""
    try:
        import settings
        return (settings.load()["dj"]["persona"]["voice"].get("refs") or {})
    except Exception:
        return {}


def decide(refs: dict[str, Any] | None = None,
           chatterbox_voice: str | None = None) -> dict[str, Any]:
    """Pure routing decision, so it can be asserted without synthesizing.

    Returns provider, the voice string that provider will be sent, and
    the one permitted fallback voice (or ''). No I/O, no network.

    `chatterbox_voice` is the ALREADY-RESOLVED asset name. dj_engine
    passes the value its long-standing resolve_tts_voice() produces, so
    the settings ref, the DJ_TTS_VOICE development override and the
    empty-means-service-default rule all keep behaving exactly as they
    did before this router existed."""
    refs = _refs() if refs is None else refs
    if chatterbox_voice is None:
        chatterbox_voice = str(refs.get(CHATTERBOX_REF) or "")
    ha_entity = str(refs.get(HA_REF) or "")
    if ha_entity:
        # An explicit Home Assistant selection is the route -- when Home
        # Assistant is configured. Otherwise the installation keeps the
        # route it already had, and /health/problems says why.
        if providers.ha_configured():
            return {"provider": PROVIDER_HOME_ASSISTANT, "voice": ha_entity,
                    "fallback_voice": ""}
        logger.warning("tts: Home Assistant voice %s is selected but Home "
                       "Assistant is not configured; using the existing route",
                       ha_entity)
    wanted = str(refs.get(KOKORO_REF) or "")
    if wanted and providers.kokoro_enabled():
        try:
            kokoro_id = kokoro_voices.kokoro_id_for(wanted)
        except kokoro_voices.UnknownVoice:
            # A voice nobody shelves is a configuration error, not a
            # reason to improvise a speaker. Fall back to the existing
            # installation rather than guessing.
            logger.warning("tts: unknown kokoro voice %r; using chatterbox",
                           wanted)
            return {"provider": PROVIDER_CHATTERBOX,
                    "voice": chatterbox_voice, "fallback_voice": ""}
        return {"provider": PROVIDER_KOKORO, "voice": kokoro_id,
                "fallback_voice": chatterbox_voice}
    return {"provider": PROVIDER_CHATTERBOX, "voice": chatterbox_voice,
            "fallback_voice": ""}


def tuning_for(voice_id: str) -> dict[str, float]:
    """The owner's delivery settings for one curated voice, as the Kokoro
    provider's keyword arguments -- only the values that differ from the
    engine's own, so an untuned voice is called exactly as before."""
    import settings
    raw = (settings.load()["dj"].get("voice_tuning") or {}).get(voice_id) or {}
    out: dict[str, float] = {}
    for field, kw in (("speed", "speed"), ("pause", "silence_scale")):
        val = raw.get(field)
        if (isinstance(val, (int, float)) and not isinstance(val, bool)
                and float(val) != 1.0):
            out[kw] = float(val)
    return out


def synthesize(text: str, *, refs: dict[str, Any] | None = None,
               chatterbox_voice: str | None = None,
               kokoro_fn: Callable[..., bytes] | None = None,
               chatterbox_fn: Callable[..., bytes] | None = None,
               ha_fn: Callable[..., bytes] | None = None) -> bytes:
    """Speak `text` with the routed provider, with at most one bounded
    fallback. Raises if speech cannot be produced at all."""
    kokoro_fn = kokoro_fn or providers.kokoro_synthesize
    chatterbox_fn = chatterbox_fn or providers.chatterbox_synthesize
    ha_fn = ha_fn or providers.ha_tts_synthesize
    refs = _refs() if refs is None else refs
    route = decide(refs, chatterbox_voice)

    if route["provider"] == PROVIDER_HOME_ASSISTANT:
        # No cross-provider fallback: a Home Assistant failure is a
        # precise, named failure (ha_tts.HAError) the break records.
        return ha_fn(text, route["voice"])

    if route["provider"] == PROVIDER_CHATTERBOX:
        return chatterbox_fn(text, route["voice"])

    try:
        tuning = tuning_for(str(refs.get(KOKORO_REF) or ""))
        return kokoro_fn(text, route["voice"], **tuning)
    except Exception as exc:                    # noqa: BLE001 - the seam
        fallback = route["fallback_voice"]
        if not fallback:
            # Nothing proven to fall back TO. Fail honestly rather than
            # putting an unintended voice on air.
            logger.warning("tts: kokoro voice %s failed and no chatterbox "
                           "fallback is configured: %s", route["voice"], exc)
            raise
        logger.warning("tts: kokoro voice %s failed (%s); falling back once "
                       "to chatterbox voice %s", route["voice"], exc, fallback)
        # ONE attempt, at the other provider. Whatever this raises is
        # final -- there is deliberately no route back to Kokoro.
        return chatterbox_fn(text, fallback)


# --- one named voice, no fallback --------------------------------------------

class VoiceUnavailable(RuntimeError):
    """A named shelved voice cannot speak right now. `reason` is a short
    code -- never a provider's own words, which may carry a URL -- so the
    caller can record WHY it fell back to another voice."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason


UNAVAILABLE_UNKNOWN = "unknown_voice"
UNAVAILABLE_HA = "home_assistant_unconfigured"
UNAVAILABLE_KOKORO = "kokoro_disabled"
UNAVAILABLE_ENGINE = "voice_engine_unavailable"
UNAVAILABLE_NO_SAMPLE = "no_sample"
UNAVAILABLE_PROVIDER = "provider_error"
UNAVAILABLE_EMPTY = "empty_audio"


def provider_for(voice_id: str) -> str:
    """Which provider a shelved voice id speaks through, from the id alone
    (no I/O); '' for an id no shelf claims."""
    import voice_library
    kind = voice_library.kind_of(voice_id)
    if kind == voice_library.HA_PROVIDER:
        return PROVIDER_HOME_ASSISTANT
    if kind == kokoro_voices.PROVIDER:
        return PROVIDER_KOKORO
    if kind == voice_library.CUSTOM_KIND:
        return PROVIDER_CHATTERBOX
    return ""


def synthesize_voice(text: str, voice_id: str, *,
                     kokoro_fn: Callable[..., bytes] | None = None,
                     chatterbox_fn: Callable[..., bytes] | None = None,
                     ha_fn: Callable[..., bytes] | None = None,
                     fetch_fn=None) -> bytes:
    """Speak `text` with ONE named shelved voice -- the voice a calendar
    announcement asked for -- or raise VoiceUnavailable.

    Deliberately NOT `synthesize`: that call routes the PERSONA's refs and
    carries the persona's one bounded fallback. Here the caller asked for
    a particular person, so there is no fallback of any kind inside this
    function: a Kokoro voice that cannot speak does not become the
    Chatterbox sample, a Home Assistant voice on an unconfigured Home
    Assistant does not become the service default. Whether, and to what,
    the ANNOUNCEMENT falls back is the announcement's decision, taken
    where it can be written down."""
    import voice_library
    kokoro_fn = kokoro_fn or providers.kokoro_synthesize
    chatterbox_fn = chatterbox_fn or providers.chatterbox_synthesize
    ha_fn = ha_fn or providers.ha_tts_synthesize
    kind = voice_library.kind_of(voice_id)
    if kind is None:
        raise VoiceUnavailable(UNAVAILABLE_UNKNOWN)
    if kind == voice_library.HA_PROVIDER:
        if not providers.ha_configured():
            raise VoiceUnavailable(UNAVAILABLE_HA)
        speak = lambda: ha_fn(text, voice_library.ha_entity_for(voice_id))  # noqa: E731
    elif kind == kokoro_voices.PROVIDER:
        if not providers.kokoro_enabled():
            raise VoiceUnavailable(UNAVAILABLE_KOKORO)
        kokoro_id = kokoro_voices.kokoro_id_for(voice_id)
        tuning = tuning_for(voice_id)
        speak = lambda: kokoro_fn(text, kokoro_id, **tuning)  # noqa: E731
    else:
        try:
            asset = voice_library.resolve_asset(voice_id, fetch_fn)
        except voice_library.VoiceEngineUnavailable as exc:
            raise VoiceUnavailable(UNAVAILABLE_ENGINE) from exc
        except voice_library.VoiceError as exc:
            raise VoiceUnavailable(UNAVAILABLE_NO_SAMPLE) from exc
        speak = lambda: chatterbox_fn(text, asset)  # noqa: E731
    try:
        audio = speak()
    except Exception as exc:  # noqa: BLE001 - the provider boundary
        logger.warning("tts: voice %s failed to speak: %s: %s", voice_id,
                       type(exc).__name__, str(exc)[:160])
        raise VoiceUnavailable(UNAVAILABLE_PROVIDER,
                               type(exc).__name__) from exc
    if not audio or len(audio) < 128:
        raise VoiceUnavailable(UNAVAILABLE_EMPTY)
    return audio


def health() -> dict[str, Any]:
    """Operational readiness of the Kokoro side only. Chatterbox keeps the
    health surfaces it already had; this adds the one signal routing and
    rollout need, and nothing else."""
    if not providers.kokoro_enabled():
        return {"enabled": False, "ok": False, "voices": 0, "error": ""}
    try:
        voices = providers.kokoro_voices()
    except Exception as exc:                    # noqa: BLE001
        return {"enabled": True, "ok": False, "voices": 0, "error": str(exc)}
    return {"enabled": True, "ok": True, "voices": len(voices), "error": ""}
