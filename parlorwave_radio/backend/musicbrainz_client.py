#!/usr/bin/env python3

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from paths import BASE_DIR

logger = logging.getLogger("musicbrain")

# Created lazily on first cache write (_save_json), never at import time.
CACHE_DIR = BASE_DIR / ".musicbrainz_cache"
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2"

# MusicBrainz requires a meaningful User-Agent with a real contact. Configure
# it with MUSICBRAINZ_CONTACT (an email or URL) where this installation
# actually talks to MusicBrainz -- catalog enrichment, which the base
# installation does not do.
#
# The warning is raised on the FIRST REQUEST, not at import. It used to run
# at module scope and print on any install that merely imported this module,
# so a fresh public add-on -- which never calls MusicBrainz at all -- opened
# its log with a warning about an optional variable for a feature it does
# not have. A warning nobody can act on teaches people to stop reading the
# log; this one now appears exactly when it is true and actionable.
CONTACT = os.environ.get("MUSICBRAINZ_CONTACT", "").strip()
PLACEHOLDER_CONTACT = "contact-not-set"
USER_AGENT = f"ParlorwaveRadio/1.0 ( {CONTACT or PLACEHOLDER_CONTACT} )"
_contact_warned = False


def _warn_unset_contact() -> None:
    """Once per process, and only from a path that really is about to call
    MusicBrainz."""
    global _contact_warned
    if CONTACT or _contact_warned:
        return
    _contact_warned = True
    logger.warning(
        "MUSICBRAINZ_CONTACT is not set; MusicBrainz is being called with a "
        "placeholder User-Agent and may throttle or refuse the request. Set "
        "it to an email address or a URL.")

# MusicBrainz allows ~1 request/second. Keep a small margin above that.
MIN_REQUEST_INTERVAL = float(os.environ.get("MUSICBRAINZ_MIN_INTERVAL", "1.1"))
_last_request_time = 0.0
_throttle_lock = threading.Lock()

# Max release-group pages (100 groups each) to fetch per artist.
MAX_RG_PAGES = 5

# Cached discographies older than this are refetched.
CACHE_TTL_DAYS = float(os.environ.get("MUSICBRAINZ_CACHE_TTL_DAYS", "30"))
CACHE_TTL_SECONDS = CACHE_TTL_DAYS * 86400

# Per-file TTL jitter (±fraction) so a batch of artists cached on the same day do
# not all expire together and storm MusicBrainz on one nightly build. The factor
# is derived deterministically from the filename, so a given file's effective TTL
# is stable across checks (it does not flip fresh/stale between reads).
CACHE_TTL_JITTER = float(os.environ.get("MUSICBRAINZ_CACHE_TTL_JITTER", "0.2"))


def _jittered_ttl(path: Path) -> float:
    if CACHE_TTL_SECONDS <= 0 or CACHE_TTL_JITTER <= 0:
        return CACHE_TTL_SECONDS
    digest = int(hashlib.sha1(path.name.encode("utf-8")).hexdigest(), 16)
    frac = (digest % 10_000) / 10_000.0  # stable 0..1 for this file
    factor = (1.0 - CACHE_TTL_JITTER) + frac * (2.0 * CACHE_TTL_JITTER)
    return CACHE_TTL_SECONDS * factor

# Cache schema version. v1 (a bare JSON list of releases, name-keyed only) is not
# migrated: v1 files are simply ignored on read, and the TTL flow repopulates them
# in the v2 format (per the hardening precedent -- derived caches are disposable).
# v2 is keyed by the artist MBID and stores the resolved artist identity alongside
# the releases; a small name-keyed pointer file records name -> MBID so that
# name->MBID resolution stays cache-warm without a fresh MusicBrainz search.
CACHE_SCHEMA_VERSION = 2

# The identity fields captured from the chosen MusicBrainz artist and embedded on
# every returned release so callers (collector, missing_albums) receive the MBID
# and canonical name alongside the discography.
IDENTITY_FIELDS = ("artist_mbid", "mb_name", "disambiguation", "mb_score")

# Minimum MusicBrainz search score (0-100) accepted for a non-exact name match.
# An exact (case-insensitive) name match is always accepted regardless of score;
# this threshold only guards the fuzzy case, replacing the old "blind artists[0]"
# pick so a low-confidence wrong artist is dropped rather than silently poisoning
# the collection data. Tunable against the real library via the env var.
ARTIST_SCORE_THRESHOLD = float(os.environ.get("MUSICBRAINZ_ARTIST_SCORE_THRESHOLD", "80"))

# Manual name -> MBID pins for the artists MusicBrainz search resolves wrongly.
# Local config (gitignored, like lidarr_config.json); optional and usually absent.
OVERRIDES_PATH = CACHE_DIR.parent / "artist_overrides.json"


_SAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+")


def _name_pointer_path(artist_name: str) -> Path:
    """Name-keyed pointer file recording the resolved artist MBID for a name."""
    normalized = "_".join(artist_name.lower().split())
    safe_name = _SAFE_NAME_RE.sub("_", normalized).strip("._") or "artist"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return CACHE_DIR / f"name_{safe_name}_{digest}.json"


def _mbid_cache_path(artist_mbid: str) -> Path:
    """MBID-keyed cache file holding the artist identity and its releases."""
    return CACHE_DIR / f"mbid_{artist_mbid}.json"


def _load_json_v2(path: Path) -> dict[str, Any] | None:
    """Read a v2 cache file if present, fresh, and correctly versioned.

    Returns None on a miss, on staleness, or for any old-format (v1 bare-list)
    file -- old-format files are ignored, never migrated.
    """
    if not path.exists():
        return None
    ttl = _jittered_ttl(path)
    if ttl > 0 and (time.time() - path.stat().st_mtime) > ttl:
        return None  # stale -> treat as a miss so the caller refetches
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if isinstance(data, dict) and data.get("v") == CACHE_SCHEMA_VERSION:
        return data
    return None  # v1 bare list or unknown format -> ignored


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _attach_identity(
    releases: list[dict[str, Any]], identity: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return releases with the artist identity fields merged onto each row."""
    fields = {k: identity.get(k) for k in IDENTITY_FIELDS}
    return [{**release, **fields} for release in releases]


def _load_cached(artist_mbid: str) -> list[dict[str, Any]] | None:
    """Return cached releases for an artist MBID (identity re-attached), or None."""
    data = _load_json_v2(_mbid_cache_path(artist_mbid))
    if data is None:
        return None
    releases = data.get("releases")
    identity = data.get("artist") or {}
    if isinstance(releases, list):
        return _attach_identity(releases, identity)
    return None


def _save_cached(
    identity: dict[str, Any], releases: list[dict[str, Any]]
) -> None:
    """Persist an artist's identity + lean releases (v2), MBID-keyed."""
    artist_mbid = identity.get("artist_mbid")
    if not artist_mbid:
        return
    _save_json(
        _mbid_cache_path(artist_mbid),
        {"v": CACHE_SCHEMA_VERSION, "artist": identity, "releases": releases},
    )


def _load_pointer(artist_name: str) -> dict[str, Any] | None:
    """Return the cached artist identity for a name via the pointer file, or None."""
    data = _load_json_v2(_name_pointer_path(artist_name))
    if data is None:
        return None
    identity = data.get("artist")
    if isinstance(identity, dict) and identity.get("artist_mbid"):
        return identity
    return None


def _save_pointer(artist_name: str, identity: dict[str, Any]) -> None:
    _save_json(
        _name_pointer_path(artist_name),
        {"v": CACHE_SCHEMA_VERSION, "artist": identity},
    )


def _throttle() -> None:
    """Block until at least MIN_REQUEST_INTERVAL has elapsed since the last call.

    Holds a process-wide lock across the sleep so concurrent threadpool requests
    are serialized and cannot collectively exceed MusicBrainz's 1 req/s policy.
    """
    global _last_request_time
    with _throttle_lock:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _request_json(url: str) -> Any:
    _warn_unset_contact()
    _throttle()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def _load_overrides() -> dict[str, str]:
    """Return {lowercased name: MBID} manual pins, or {} if none/unreadable."""
    try:
        with open(OVERRIDES_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(name).strip().lower(): str(mbid).strip()
        for name, mbid in data.items()
        if str(name).strip() and str(mbid).strip()
    }


def _score_of(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _select_artist(
    artist_name: str, artists: list[Any], threshold: float = ARTIST_SCORE_THRESHOLD
) -> dict[str, Any] | None:
    """Choose the MusicBrainz artist for a name.

    An exact (case-insensitive) name match is accepted regardless of score. With
    no exact match, the highest-scoring candidate is accepted only if its MB
    search score clears ``threshold``; otherwise nothing is chosen (the name is
    left unresolved rather than pinned to a low-confidence wrong artist).
    """
    ranked = [c for c in artists if isinstance(c, dict)]
    if not ranked:
        return None
    for candidate in ranked:
        if str(candidate.get("name", "")).lower() == artist_name.lower():
            return candidate
    best = max(ranked, key=_score_of)
    return best if _score_of(best) >= threshold else None


def _identity_from(artist: dict[str, Any], fallback_name: str, score: Any) -> dict[str, Any]:
    return {
        "artist_mbid": artist.get("id"),
        "mb_name": (artist.get("name") or "").strip() or fallback_name,
        "disambiguation": (artist.get("disambiguation") or "").strip(),
        "mb_score": score,
    }


def _fetch_identity_by_mbid(mbid: str, fallback_name: str) -> dict[str, Any]:
    """Resolve a pinned MBID to a full identity via a direct artist lookup.

    Falls back to a minimal identity (just the pinned MBID) if the lookup fails,
    so an override always resolves. mb_score is None -- an override is not scored.
    """
    url = f"{MUSICBRAINZ_URL}/artist/{urllib.parse.quote(mbid)}?fmt=json"
    try:
        data = _request_json(url)
    except Exception:
        data = None
    if isinstance(data, dict) and data.get("id"):
        return _identity_from(data, fallback_name, None)
    return {"artist_mbid": mbid, "mb_name": fallback_name, "disambiguation": "", "mb_score": None}


def _resolve_identity(artist_name: str) -> dict[str, Any] | None:
    """Resolve a name to the chosen artist's identity (override, else search).

    Identity = MBID, canonical MB name, disambiguation, and MB search score.
    Returns None if nothing clears the confidence threshold.
    """
    pinned = _load_overrides().get(artist_name.strip().lower())
    if pinned:
        return _fetch_identity_by_mbid(pinned, artist_name)

    query = urllib.parse.quote(artist_name)
    url = f"{MUSICBRAINZ_URL}/artist/?query=artist:{query}&fmt=json&limit=5"
    try:
        data = _request_json(url)
    except Exception:
        return None

    artists = data.get("artists", []) if isinstance(data, dict) else []
    artist = _select_artist(artist_name, artists)
    if artist is None or not artist.get("id"):
        return None
    return _identity_from(artist, artist_name, artist.get("score"))


def lookup_artist_release_groups(artist_name: str) -> list[dict[str, Any]]:
    """Return an artist's studio-eligible release groups from MusicBrainz.

    Each returned release carries the resolved artist identity fields
    (artist_mbid, mb_name, disambiguation, mb_score) so callers get the MBID
    alongside the discography. Resolution and releases are cached separately
    (name pointer + MBID-keyed release cache) so a warm name->MBID pointer avoids
    a fresh search even when the release cache has expired.
    """
    # Resolve identity: prefer a warm name pointer, else search MusicBrainz.
    identity = _load_pointer(artist_name)
    if identity is None:
        identity = _resolve_identity(artist_name)
        if identity is None:
            return []
        _save_pointer(artist_name, identity)

    artist_id = identity["artist_mbid"]

    cached = _load_cached(artist_id)
    if cached is not None:
        return cached

    groups: list[dict[str, Any]] = []
    offset = 0
    complete = False
    for _ in range(MAX_RG_PAGES):
        release_groups_url = (
            f"{MUSICBRAINZ_URL}/release-group?artist={artist_id}"
            f"&fmt=json&limit=100&offset={offset}"
        )
        try:
            release_data = _request_json(release_groups_url)
        except Exception:
            break
        if not isinstance(release_data, dict):
            break
        page = [g for g in (release_data.get("release-groups") or []) if isinstance(g, dict)]
        groups.extend(page)
        total = release_data.get("release-group-count")
        offset += len(page)
        if len(page) < 100 or (isinstance(total, int) and offset >= total):
            complete = True
            break
    if not groups:
        return []

    releases = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        title = (group.get("title") or "").strip()
        if not title:
            continue
        first_date = (group.get("first-release-date") or "")[:4]
        year = int(first_date) if first_date.isdigit() else 0
        primary_type = str(group.get("primary-type") or "").strip()
        secondary_types = [str(item).strip() for item in (group.get("secondary-types") or []) if str(item).strip()]
        release_group_type = ""
        if primary_type and primary_type.lower() == "album" and secondary_types:
            release_group_type = secondary_types[0]
        elif primary_type:
            release_group_type = primary_type
        elif secondary_types:
            release_group_type = secondary_types[0]
        releases.append({
            "artist": artist_name,
            "album": title,
            "year": year,
            "musicbrainz_release_group_id": group.get("id"),
            "release_group_type": release_group_type,
            "primary_type": primary_type,
            "secondary_types": secondary_types,
        })

    if complete:
        _save_cached(identity, releases)  # lean releases stored MBID-keyed
    return _attach_identity(releases, identity)
