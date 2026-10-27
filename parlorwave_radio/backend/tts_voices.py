"""The DJ voice catalog: what a settings UI may offer and what a settings
write may persist as dj.persona.voice.refs.chatterbox_voice.

Voice is DELIVERY only. Changing it changes what synthesizes the NEXT
rendered break; it never touches the prompt, the programming, the queue or
the player. The stored value is the TTS service's own voice name (the value
the speech request carries), because that is the one identifier a
synthesizer round-trips; the UI shows a friendly name derived from it and
never a file path.

The empty id means "the service's default voice": the speech request omits
the voice key entirely (providers.chatterbox_synthesize), exactly as an
installation with no voice configured has always behaved.

The inventory is read live from the service and cached briefly. A write
naming a voice the service does not list is refused; a write while the
inventory cannot be read is refused too (503, not 422) -- persisting an
unverifiable voice would fail every break until someone noticed.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import providers

DEFAULT_VOICE_ID = ""
DEFAULT_VOICE_NAME = "Service default"
CACHE_TTL_S = 60.0

_cache: dict[str, Any] = {"at": 0.0, "voices": None}


class VoiceCatalogUnavailable(RuntimeError):
    """The service could not confirm which voices exist right now."""


def friendly_name(asset: str) -> str:
    """legacy_voice -> 'Legacy Voice'. The asset name is the installation's own
    label; we tidy it, we do not rename it."""
    words = [w for w in str(asset or "").replace("-", " ").replace("_", " ")
             .split() if w]
    return " ".join(w[:1].upper() + w[1:] for w in words) or str(asset)


def inventory(fetch_fn: Callable[[], list[dict[str, Any]]] | None = None,
              now: float | None = None) -> list[str]:
    """Live voice ids (service names), cached CACHE_TTL_S. Raises
    VoiceCatalogUnavailable when the service cannot answer and nothing
    fresh enough is cached."""
    t = time.monotonic() if now is None else now
    if _cache["voices"] is not None and t - _cache["at"] < CACHE_TTL_S:
        return list(_cache["voices"])
    fetch = fetch_fn or providers.chatterbox_voices
    try:
        raw = fetch()
    except Exception as exc:  # noqa: BLE001 - any failure = unknown inventory
        raise VoiceCatalogUnavailable(str(exc)[:200]) from exc
    ids: list[str] = []
    for rec in raw:
        name = str(rec.get("name") or "")
        if name and name not in ids:
            ids.append(name)
    _cache.update(at=t, voices=ids)
    return list(ids)


def reset_cache() -> None:
    _cache.update(at=0.0, voices=None)


def catalog(fetch_fn: Callable[[], list[dict[str, Any]]] | None = None,
            now: float | None = None) -> dict[str, Any]:
    """What a Studio may show. Always answers: an unreadable service yields
    ok=False with the default entry only, so a UI can still render the
    current selection and say the list could not be loaded."""
    entries = [{"id": DEFAULT_VOICE_ID, "name": DEFAULT_VOICE_NAME}]
    try:
        ids = inventory(fetch_fn, now)
    except VoiceCatalogUnavailable as exc:
        return {"ok": False, "voices": entries, "error": str(exc)}
    entries.extend({"id": v, "name": friendly_name(v)} for v in ids)
    return {"ok": True, "voices": entries, "error": None}


def validate_choice(voice: Any,
                    fetch_fn: Callable[[], list[dict[str, Any]]] | None = None,
                    now: float | None = None) -> str:
    """The voice id a settings write may persist, or ValueError (unknown
    voice) / VoiceCatalogUnavailable (cannot verify right now)."""
    if voice is None or voice == DEFAULT_VOICE_ID:
        return DEFAULT_VOICE_ID
    if not isinstance(voice, str):
        raise ValueError("voice must be a string")
    if voice not in inventory(fetch_fn, now):
        raise ValueError(f"voice '{voice}' is not offered by the TTS service")
    return voice
