#!/usr/bin/env python3
"""Canonical media identity, separate from catalog membership.

Two questions that the old model answered with one column:

  A. MEDIA IDENTITY  -- what item is actually playing?
  B. CATALOG MATCH   -- does that item correspond to a local library track?

`session_tracks.path` answered both, so anything without B could not be
represented at all: an unmatched item produced no row, no history, nothing. A
must be able to exist without B.

IDENTITY COMES FROM THE SOURCE, NOT FROM METADATA. Home Assistant already
reports `media_content_id` for Music Assistant playback -- `library://track/4821`
for a library item, `spotify://track/...` for a provider item. That IS the
stable identifier, so it is what we key on. Deliberately NOT used as identity:

  - title+artist, which is not identity (two different masters, live versions,
    re-recordings and covers all collide, and a rename silently forks history)
  - a hash of metadata, which is the same collision dressed up as a key and
    would only exist to avoid carrying the real identifier

When no content_id is reported the item still gets a session row -- it is real
playback -- but it has no stable cross-session identity, and says so.
"""

from __future__ import annotations

from typing import Any

# match_status describes RESOLUTION STATE, never provenance.
#
# The distinction is not academic: a local library often holds several rows
# with one title and artist (the album cut, a compilation, a live record), so
# a catalog lookup by title+artist can fail for a track that is plainly LOCAL
# (library://track/N). Calling such an item "external" would be simply false.
# Whether we resolved an item and where it came from are different questions,
# and only the first is one we can answer.
CATALOG = "catalog"          # resolved to a local library track
UNRESOLVED = "unresolved"    # stable media identity, no catalog match (yet)
UNIDENTIFIED = "unidentified"  # no identity and no catalog match; never stored

# Local files are keyed by their own namespace so a catalog path can never
# collide with a provider URI.
FILE_SCHEME = "file://"


def _split_scheme(uri: str) -> tuple[str, str]:
    if "://" not in uri:
        return "", uri
    scheme, rest = uri.split("://", 1)
    return scheme.lower(), rest


def namespace_of(media_content_id: str | None) -> str:
    """The IDENTITY NAMESPACE -- the URI scheme.

    Deliberately not called "provider". `library://track/7316` is Music
    Assistant's library namespace; the actual content provider for that item is
    filesystem_smb--Ab12Cd34, which lives in MA's streamdetails and is NOT
    exposed to Home Assistant. Naming this field "provider" would claim
    provenance we were never given.
    """
    uri = (media_content_id or "").strip()
    if not uri:
        return ""
    scheme, _ = _split_scheme(uri)
    return scheme or "unknown"


def provider_item_id(media_content_id: str | None) -> str:
    uri = (media_content_id or "").strip()
    if not uri:
        return ""
    _, rest = _split_scheme(uri)
    return rest


def media_key(media_content_id: str | None, catalog_path: str | None = None) -> str:
    """Stable identity for this item, or '' when the source gave us none.

    A provider URI wins over the catalog path: it is the identity the player
    reported, it survives a library re-index, and it is what makes the same
    item recognisable next time. The catalog path is the fallback so
    library-only flows (the DJ's own queue) keep a stable key too.
    """
    uri = (media_content_id or "").strip()
    if uri:
        scheme, rest = _split_scheme(uri)
        return f"{scheme}://{rest}" if scheme else uri
    path = (catalog_path or "").strip()
    return f"{FILE_SCHEME}{path}" if path else ""


def describe(payload: dict[str, Any],
             catalog_path: str | None = None) -> dict[str, Any]:
    """The canonical reference for one playing item.

    catalog_path is whatever resolution independently determined, or None. It
    is never invented here: no catalog path means no catalog path.
    """
    content_id = (payload.get("media_content_id") or "").strip()
    path = (catalog_path or "").strip()
    key = media_key(content_id, path)
    if path:
        status = CATALOG
    elif key:
        status = UNRESOLVED
    else:
        status = UNIDENTIFIED
    return {
        "media_key": key,
        "media_uri": content_id,
        "media_namespace": namespace_of(content_id),
        "provider_item_id": provider_item_id(content_id),
        "catalog_path": path,
        "match_status": status,
        "title": (payload.get("media_title") or "").strip(),
        "artist": (payload.get("media_artist") or "").strip(),
        # Album only when the player supplied it: an unresolved item must not
        # acquire library metadata it never had. Home Assistant does expose
        # media_album_name, so this is real data, not a placeholder.
        "album": (payload.get("media_album_name") or "").strip(),
        "duration": payload.get("duration"),
    }


# Finite PROGRAMMING media: audio Music Brain itself put in the queue, which
# is never music however the player labels it. Each entry is a rooted path
# segment matched on the URI PATH only, so a song legitimately titled
# "Broadcast" or living in a News folder can never trip it.
BREAK_AUDIO_SEGMENT = "/broadcast/audio/"
NEWS_AUDIO_SEGMENT = "/news/audio/"

PROGRAMMING_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("break", BREAK_AUDIO_SEGMENT),   # the station's own voice
    ("news", NEWS_AUDIO_SEGMENT),     # a publisher bulletin we cached
)


def _token_after(path_part: str, segment: str) -> str:
    idx = path_part.find(segment)
    if idx < 0:
        return ""
    token = path_part[idx + len(segment):].split("/", 1)[0]
    return token[:-4] if token.endswith(".mp3") else token


def programming_ref(media_content_id: str | None) -> tuple[str, str]:
    """(kind, token) when this media_content_id is finite programming audio
    served by Music Brain, else ('', '').

    ONE canonical identity check for every consumer that must keep
    programming out of music: adoption, session tracks, play_events, taste,
    recent-music context and the DJ writer's history. Music Assistant wraps
    our URL (builtin://track/http://host/news/audio/<token>.mp3), so the
    match is a path-segment search rather than a prefix."""
    uri = (media_content_id or "").strip()
    if not uri:
        return "", ""
    path_part = uri.split("?", 1)[0].split("#", 1)[0]
    for kind, segment in PROGRAMMING_SEGMENTS:
        token = _token_after(path_part, segment)
        if token:
            return kind, token
    return "", ""


def break_token_of(media_content_id: str | None) -> str:
    """The DJ break token when this media_content_id is a ParlorWave break audio
    URL, else ''. A NEWS bulletin is programming too, but not a break: it
    deliberately returns '' here so break lifecycle hooks never fire for it
    (see programming_ref for the general classifier)."""
    kind, token = programming_ref(media_content_id)
    return token if kind == "break" else ""


def is_catalog(ref: dict[str, Any]) -> bool:
    return ref.get("match_status") == CATALOG
