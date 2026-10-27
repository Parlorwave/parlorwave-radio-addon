"""Source-capability model.

ONE centralized place that answers "what kind of music source is this, and who
is its authority" -- so provider-name conditionals never scatter through the
codebase. Identity is read from the deterministic MA facts we already fetch
(the queue item's provider + URI prefix, and the queue's is_dynamic flag). No
fuzzy matching; a provider prefix is an exact signal.

Three item provenances Music Brain must tell apart, and their signals
(verified against MA 2.11):

    local_owned_music      provider `filesystem_*`   / uri `library://track/...`
    pandora_provider_music provider `pandora*`       / uri `pandora*://track/...`
    jarvis_program_element provider `builtin`        / uri `builtin://track/<url>`
                           (a ParlorWave element is a builtin item whose URL is a
                           Music Brain broadcast/news render)

Two source MODES a queue can be in, and who owns the music in each:

    LOCAL                  music authority = ParlorWave (ownership/ordering/
                           artist-policy apply)
    PANDORA_DYNAMIC_RADIO  music authority = the provider; Pandora chooses,
                           orders and refills; ParlorWave hosts the SHOW around it
                           (program elements + explicit local requests) but
                           never reorders Pandora's selections.

This module holds NO transport or network logic -- it classifies dicts the
caller already has. Transport ownership is a SEPARATE concern from music
authority (see queue_ownership); a ParlorWave-owned session can host a
provider-authoritative queue.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- source kinds / item provenances -----------------------------------------

LOCAL = "local"
PANDORA_DYNAMIC_RADIO = "pandora_dynamic_radio"
JARVIS_PROGRAM = "jarvis_program"
UNKNOWN = "unknown"

# item-provenance labels used by telemetry / the future provider-agnostic seam
LOCAL_OWNED_MUSIC = "local_owned_music"
PANDORA_PROVIDER_MUSIC = "pandora_provider_music"
# RETAINED ON PURPOSE, and the only two retired-brand strings left in
# shipped runtime code. These are PROTOCOL VALUES, not words: they are
# written into the queue-ownership table, compared against rows already on
# disk in every existing installation, and exchanged between the observer
# and the station resolver. Renaming them is a data migration with nothing
# to gain -- no customer ever sees either string, which the release scan
# asserts rather than assumes.
JARVIS_PROGRAM_ELEMENT = "jarvis_program_element"

# music-authority values
AUTHORITY_JARVIS = "jarvis"
AUTHORITY_PROVIDER = "provider"
AUTHORITY_UNKNOWN = "unknown"

# Exact provider/URI identity signals (prefix match only -- never fuzzy).
_PANDORA_PREFIX = "pandora"
_BUILTIN_PROVIDER = "builtin"
_LOCAL_PROVIDER_PREFIXES = ("filesystem_", "library")
_LOCAL_URI_PREFIXES = ("library://track", "library://album")
_BUILTIN_URI_PREFIX = ("builtin://track/", "builtin://sound_effect/")


@dataclass(frozen=True)
class SourceCapabilities:
    """What a source mode allows. The fields the Pandora product rules turn on."""

    kind: str
    music_authority: str            # AUTHORITY_JARVIS | AUTHORITY_PROVIDER
    exact_track_selectable: bool     # can we play an arbitrary exact track on demand?
    reorderable: bool                # may ParlorWave reorder this source's music?
    source_managed_refill: bool      # does the source refill itself (vs ParlorWave)?
    upcoming_track_visibility: bool  # are upcoming items discrete/visible?
    foreign_item_interleave: bool    # can we insert a foreign finite item next?
    station_selectable: bool = False  # can an existing station be selected?


_PROFILES: dict[str, SourceCapabilities] = {
    LOCAL: SourceCapabilities(
        kind=LOCAL,
        music_authority=AUTHORITY_JARVIS,
        exact_track_selectable=True,
        reorderable=True,
        source_managed_refill=False,
        upcoming_track_visibility=True,
        foreign_item_interleave=True,
        station_selectable=False,
    ),
    PANDORA_DYNAMIC_RADIO: SourceCapabilities(
        kind=PANDORA_DYNAMIC_RADIO,
        music_authority=AUTHORITY_PROVIDER,
        exact_track_selectable=False,
        reorderable=False,
        source_managed_refill=True,
        upcoming_track_visibility=True,
        foreign_item_interleave=True,
        station_selectable=True,
    ),
}


def capabilities_for(kind: str) -> SourceCapabilities | None:
    """The capability profile for a source mode, or None for an unknown kind."""
    return _PROFILES.get(kind)


# --- item classification ------------------------------------------------------

def _provider_of(item: dict) -> str:
    sd = item.get("streamdetails") or {}
    mi = item.get("media_item") or {}
    return str(sd.get("provider") or mi.get("provider")
               or item.get("provider") or "")


def _uri_of(item: dict) -> str:
    mi = item.get("media_item") or {}
    # `media_uri` is the catalog-track spelling; `uri` the MA queue-item spelling
    return str(item.get("uri") or mi.get("uri") or item.get("media_uri") or "")


def item_provenance(item: dict) -> str:
    """Classify ONE queue item as local_owned / pandora_provider / jarvis_program
    / unknown, from its provider prefix (URI as a fallback signal)."""
    prov = _provider_of(item)
    uri = _uri_of(item)
    if prov.startswith(_PANDORA_PREFIX) or uri.startswith(_PANDORA_PREFIX):
        return PANDORA_PROVIDER_MUSIC
    if prov == _BUILTIN_PROVIDER or uri.startswith(_BUILTIN_URI_PREFIX):
        return JARVIS_PROGRAM_ELEMENT
    if prov.startswith(_LOCAL_PROVIDER_PREFIXES) or uri.startswith(_LOCAL_URI_PREFIXES):
        return LOCAL_OWNED_MUSIC
    return UNKNOWN


def is_pandora_item(item: dict) -> bool:
    return item_provenance(item) == PANDORA_PROVIDER_MUSIC


def is_jarvis_program_item(item: dict) -> bool:
    return item_provenance(item) == JARVIS_PROGRAM_ELEMENT


# --- queue-mode classification ------------------------------------------------

def _queue_is_dynamic(queue_state: dict) -> bool:
    return bool(queue_state.get("is_dynamic"))


def _queue_has_pandora(queue_state: dict) -> bool:
    """True when the queue's own source or its current item is Pandora."""
    for s in (queue_state.get("sources") or ()):
        uri = str(s.get("uri") or "")
        # a Pandora station source surfaces as library://radio/... backed by the
        # pandora provider; the current OR next item's provider is the tell.
        if uri.startswith(_PANDORA_PREFIX):
            return True
    return (is_pandora_item(queue_state.get("current_item") or {})
            or is_pandora_item(queue_state.get("next_item") or {}))


def queue_mode(queue_state: dict) -> str:
    """The music-authority MODE of the SESSION, not of the item currently
    playing.

    Invariant: a Pandora dynamic queue stays provider-authority even while
    an inserted LOCAL request or a ParlorWave program element is the current item.
    So the authoritative signal is the QUEUE's own dynamic identity
    (`is_dynamic`), which does NOT change when a finite foreign item is spliced
    in -- NOT the current item's provenance.

      is_dynamic True   -> PANDORA_DYNAMIC_RADIO (the only dynamic source we
                           host; provider owns selection/order/refill)
      is_dynamic False  -> LOCAL (an explicit ParlorWave/local session)
      is_dynamic absent -> prove it from a visible Pandora item (current/next/
                           source); if that cannot be proven, UNKNOWN -- never
                           silently switch a possibly-Pandora session to local
                           just because a local request happens to be playing.
    """
    dyn = queue_state.get("is_dynamic")
    if dyn is True:
        return PANDORA_DYNAMIC_RADIO
    if dyn is False:
        return LOCAL
    if _queue_has_pandora(queue_state):
        return PANDORA_DYNAMIC_RADIO
    return UNKNOWN


def music_authority(queue_state: dict) -> str:
    """Who owns selection/order/refill for this queue's music. UNKNOWN mode
    yields AUTHORITY_UNKNOWN -- callers must not assume ParlorWave authority."""
    mode = queue_mode(queue_state)
    if mode == UNKNOWN:
        return AUTHORITY_UNKNOWN
    return _PROFILES[mode].music_authority


def capabilities_of_queue(queue_state: dict) -> SourceCapabilities | None:
    """The capability profile for a live queue; None when the mode is UNKNOWN
    (the caller must treat an unknown source conservatively)."""
    return _PROFILES.get(queue_mode(queue_state))


# --- observability + DJ runway ------------------------------------------------

def source_status(queue_state: dict) -> dict:
    """Bounded, credential-free source status for telemetry. Never includes
    tokens or signed playback URLs -- only the mode facts."""
    mode = queue_mode(queue_state)
    caps = _PROFILES.get(mode)
    ci = queue_state.get("current_item") or {}
    return {
        "source_mode": mode,
        "music_authority": caps.music_authority if caps else AUTHORITY_UNKNOWN,
        "source_managed_refill": caps.source_managed_refill if caps else None,
        "reorderable": caps.reorderable if caps else None,
        "is_dynamic": queue_state.get("is_dynamic"),
        "current_provenance": item_provenance(ci) if ci else None,
    }


def _music_view(item: dict, position: int) -> dict:
    """A provider-agnostic view of one music item for DJ context. Carries only
    fields MA actually exposed; a provider (e.g. Pandora) gives NO lyrics,
    MBID, release year, genre or trivia unless a field is literally present, so
    downstream semantics must gate on real data, never on the source."""
    mi = item.get("media_item") or {}
    album = mi.get("album") or item.get("album") or {}
    arts = mi.get("artists") or item.get("artists") or []
    artist = (", ".join(a.get("name") for a in arts if a.get("name"))
              or item.get("artist"))
    return {
        "position": position,
        "provenance": item_provenance(item),
        "queue_item_id": item.get("queue_item_id"),
        "artist": artist,
        "title": mi.get("name") or item.get("name") or item.get("title"),
        "album": album.get("name") if isinstance(album, dict) else album,
        "duration": item.get("duration") or mi.get("duration"),
    }


def is_program_item(item: dict) -> bool:
    """Is this queue item finite Parlorwave programming rather than music?

    TWO independent signals, either of which is enough, because they fail in
    different directions. `item_provenance` reads the provider/URI PREFIX
    (`builtin`), which is how Music Assistant labels our inserted audio once
    it has wrapped it. `media_ref.programming_ref` reads the route SEGMENT
    (`/broadcast/audio/`, `/news/audio/`) anywhere in the URI, which is the
    canonical programming check the rest of the codebase uses and which
    still recognises our audio when the builtin wrapper is absent.

    Taking either signal makes this fail CLOSED: a break that slipped past
    one check is still kept out of the music. Calling a song programming
    would only shorten a runway; calling a break music puts "Parlorwave
    Radio - DJ break" in front of the DJ as the next song, which is the
    failure this guards."""
    if item_provenance(item) == JARVIS_PROGRAM_ELEMENT:
        return True
    try:
        import media_ref
        mi = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
        uri = (mi.get("uri") or item.get("uri") or item.get("media_uri") or "")
        return bool(media_ref.programming_ref(str(uri))[0])
    except (AttributeError, TypeError, ValueError):
        return False


def music_runway(items: list[dict], current_queue_item_id: str | None = None, *,
                 back: int = 3, ahead: int = 4,
                 base_position: int = 0) -> dict:
    """The ordered MUSIC runway around the current item, for DJ context.

    Program elements (ParlorWave breaks/news) are excluded from the MUSIC runway
    -- they are the show's furniture, not the music the DJ talks about.
    `items` is an ordered SLICE of the queue (list-order is the only
    positional truth) and `base_position` is the queue position of items[0],
    so reported positions are absolute rather than window-relative.

    ANCHORING. The current item is located by `current_queue_item_id`. When
    that id is given and is NOT in the window, this returns an EMPTY,
    UNANCHORED runway -- never a runway silently rooted at position 0. A
    window read from the wrong place describes the start of the queue as
    though it were the music around the listener; saying nothing is the
    honest answer, and the DJ's grounded ending/next facts are unaffected.
    Passing no id keeps the old meaning: the caller is asserting that
    items[0] IS the current item.

    Returns {current, recent, upcoming, anchored}."""
    cur_pos = None
    if current_queue_item_id:
        for i, t in enumerate(items):
            if t.get("queue_item_id") == current_queue_item_id:
                cur_pos = i
                break
        if cur_pos is None:
            return {"current": None, "recent": [], "upcoming": [],
                    "anchored": False}
    else:
        cur_pos = 0
    music = [(i, t) for i, t in enumerate(items) if not is_program_item(t)]
    def _view(i, t):
        return _music_view(t, base_position + i)
    current = next((_view(i, t) for i, t in music if i == cur_pos), None)
    recent = [_view(i, t) for i, t in music if i < cur_pos][-back:]
    upcoming = [_view(i, t) for i, t in music if i > cur_pos][:ahead]
    return {"current": current, "recent": recent, "upcoming": upcoming,
            "anchored": True}
