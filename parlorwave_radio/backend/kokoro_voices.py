"""The seven curated built-in DJ voices (Kokoro).

Voice is WHO, personality is HOW. Nothing here knows about shifts,
schedules, chattiness, mood or the station programmer, and nothing here
may learn about them: a voice is a sound, not a character.

Three identities per voice, deliberately kept apart:

  voice_id     the stable canonical PRODUCT id. This is what settings
               persist and what an API accepts. It never changes.
  kokoro_id    the engine's own voice name (bm_lewis...), sent to the
               Kokoro service as "voice".
  speaker_id   the raw numeric speaker in the model's voice table. An
               INTERNAL implementation detail: never customer-facing,
               never accepted from an API, never persisted in settings.
               It lives here only so the service can resolve it and a
               smoke can assert the engine spoke as the right speaker.

The mapping mirrors the engine's own voice table exactly, one row per
curated voice (voice_id, kokoro_id, speaker_id). Changing a speaker_id silently
re-casts the station, so these are pinned by test.
"""

from __future__ import annotations

from typing import Any

PROVIDER = "kokoro"

#: voice_id -> canonical record. Order is the product's shelf order.
BUILTINS: tuple[dict[str, Any], ...] = (
    {"voice_id": "lewis", "label": "Lewis",
     "kokoro_id": "bm_lewis", "speaker_id": 27, "language": "en"},
    {"voice_id": "emma", "label": "Emma",
     "kokoro_id": "bf_emma", "speaker_id": 21, "language": "en"},
    {"voice_id": "puck", "label": "Puck",
     "kokoro_id": "am_puck", "speaker_id": 18, "language": "en-us"},
    {"voice_id": "michael", "label": "Michael",
     "kokoro_id": "am_michael", "speaker_id": 16, "language": "en-us"},
    {"voice_id": "george", "label": "George",
     "kokoro_id": "bm_george", "speaker_id": 26, "language": "en"},
    {"voice_id": "heart", "label": "Heart",
     "kokoro_id": "af_heart", "speaker_id": 3, "language": "en-us"},
    # The one voice whose delivery, not just its timbre, is the product
    # promise -- the label says so out loud.
    {"voice_id": "nicole", "label": "Nicole — Whisper / Late Night",
     "kokoro_id": "af_nicole", "speaker_id": 6, "language": "en-us"},
)

VOICE_IDS: tuple[str, ...] = tuple(v["voice_id"] for v in BUILTINS)
KOKORO_IDS: tuple[str, ...] = tuple(v["kokoro_id"] for v in BUILTINS)

_BY_VOICE_ID = {v["voice_id"]: v for v in BUILTINS}
_BY_KOKORO_ID = {v["kokoro_id"]: v for v in BUILTINS}


class UnknownVoice(ValueError):
    """An id no curated built-in claims. Never guess a near match: a
    wrong guess puts the wrong person on air."""


def by_voice_id(voice_id: str) -> dict[str, Any]:
    try:
        return _BY_VOICE_ID[voice_id]
    except KeyError:
        raise UnknownVoice(f"unknown built-in voice: {voice_id!r}") from None


def by_kokoro_id(kokoro_id: str) -> dict[str, Any]:
    try:
        return _BY_KOKORO_ID[kokoro_id]
    except KeyError:
        raise UnknownVoice(f"unknown kokoro voice: {kokoro_id!r}") from None


def is_builtin(voice_id: str) -> bool:
    return voice_id in _BY_VOICE_ID


def kokoro_id_for(voice_id: str) -> str:
    """Product id -> the engine's voice name. Raises for anything else,
    so an unknown voice fails loudly here rather than silently speaking
    as somebody."""
    return by_voice_id(voice_id)["kokoro_id"]


def speaker_id_for(voice_id: str) -> int:
    """INTERNAL. For the service and for smoke assertions only."""
    return by_voice_id(voice_id)["speaker_id"]


def shelf() -> list[dict[str, str]]:
    """The customer-facing view: no speaker numbers."""
    return [{"voice_id": v["voice_id"], "label": v["label"],
             "kokoro_id": v["kokoro_id"], "language": v["language"]}
            for v in BUILTINS]
