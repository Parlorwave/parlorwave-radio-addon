#!/usr/bin/env python3

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import providers

from musicbrainz_client import lookup_artist_release_groups
from release_filters import normalize_title, studio_album_reject_reason

logger = logging.getLogger("musicbrain")

from paths import BASE_DIR

INDEX = BASE_DIR / "music_index.json"
# HTTP mechanics + the authoritative endpoint/model defaults live in
# providers.py.
MODEL = providers.OLLAMA_MODEL

# Max artists listed in the library summary sent to the LLM, so the prompt does
# not grow without bound with the library. At/under this count all are listed
# (unchanged behavior for small libraries).
MAX_SUMMARY_ARTISTS = int(os.environ.get("MISSING_ALBUMS_MAX_ARTISTS", "100"))

# Initial studio albums taken per artist (favours artist diversity). If the
# final list falls short of the requested count, additional qualifying studio
# albums from already-approved artists are backfilled beyond this cap.
PER_ARTIST_CAP = 4

# Backfill only pulls in albums scoring at least this much, so padding toward the
# requested count stays era-appropriate (in-era=100, near-era +/-3yr=83,
# out-of-era=63). Keeps the quality threshold when digging deeper into a catalog.
MIN_BACKFILL_SCORE = 80

# Studio-album classification (JUNK_TYPES / JUNK_TITLE_TOKENS / studio_album
# checks) now lives in release_filters so Collector Intelligence can share it.

# Decade tokens -> preferred year window. The 90s window is deliberately
# widened to 1988-2000: a policy choice, configurable by editing this table.
DECADE_WINDOWS = [
    (("60s", "1960s", "sixties"), (1960, 1969)),
    (("70s", "1970s", "seventies"), (1970, 1979)),
    (("80s", "1980s", "eighties"), (1980, 1989)),
    (("90s", "1990s", "nineties"), (1988, 2000)),
    (("2000s", "aughts", "noughties"), (2000, 2009)),
    (("2010s",), (2010, 2019)),
]

# Prompt tokens that opt in to modern (2020+) releases.
NEW_RELEASE_TOKENS = [
    "new release", "new releases", "latest", "recent",
    "2020s", "new music", "brand new",
]


def _detect_target_window(prompt: str) -> tuple[int, int] | None:
    """Infer a preferred (start_year, end_year) window from decade hints."""
    text = normalize_title(prompt)
    for tokens, window in DECADE_WINDOWS:
        for token in tokens:
            if re.search(rf"\b{re.escape(token)}\b", text):
                return window
    return None


def _wants_new_releases(prompt: str) -> bool:
    text = normalize_title(prompt)
    return any(token in text for token in NEW_RELEASE_TOKENS)


def _year_center(target_window: tuple[int, int] | None) -> int:
    return (target_window[0] + target_window[1]) // 2 if target_window else 1994


def _year_distance(release: dict[str, Any], target_window: tuple[int, int] | None) -> int:
    year = int(release.get("year") or 0)
    return abs(year - _year_center(target_window)) if year else 9999


def build_library_summary(library: dict[str, Any], max_artists: int = MAX_SUMMARY_ARTISTS) -> str:
    artists = library.get("artists", [])
    if not artists:
        return "Library contains no artists yet."

    shown = artists[:max_artists]
    lines = ["Current library summary:"]
    for artist in shown:
        albums = artist.get("albums", []) or []
        album_text = ", ".join(albums[:8])
        if len(albums) > 8:
            album_text += f", +{len(albums) - 8} more"
        lines.append(f"- {artist.get('name', 'Unknown')}: {album_text}")
    if len(artists) > max_artists:
        lines.append(f"...and {len(artists) - max_artists} more artists")
    return "\n".join(lines)


def filter_suggestions(suggestions: list[dict[str, Any]], owned: set[tuple[str, str]]) -> list[dict[str, Any]]:
    filtered = []
    seen = set()
    for suggestion in suggestions:
        artist = (suggestion.get("artist") or "").strip()
        album = (suggestion.get("album") or "").strip()
        if not artist or not album:
            continue
        key = (artist.lower(), album.lower())
        if key in seen:
            continue
        seen.add(key)
        if (artist, album) in owned:
            continue
        filtered.append(suggestion)
    return filtered


def _reject_reason(
    release: dict[str, Any],
    target_window: tuple[int, int] | None,
    wants_new: bool,
) -> str | None:
    """Return a rejection category (or None to keep) for a release group.

    Categories: ``type`` (live/comp/EP/single/etc.), ``title`` (deluxe/
    anniversary/greatest hits/...), ``year_2020plus`` and ``non_album``.

    Layers the era/recency policy on top of the shared studio-album test:
    junk type/title always rejects; otherwise 2020+ is rejected unless the user
    asked for new releases; ambiguous-type releases inside the target era get the
    benefit of the doubt.
    """
    year = int(release.get("year") or 0)
    base = studio_album_reject_reason(release)  # "type:*" | "title" | "non_album" | None

    if base is not None and base != "non_album":
        return base  # junk type/title rejects regardless of year

    if not wants_new and year >= 2020:
        return "year_2020plus"

    if base is None:
        return None  # resolved studio album

    # base == "non_album": ambiguous type; keep only if inside the target era.
    if target_window and year and target_window[0] <= year <= target_window[1]:
        return None
    return "non_album"


def _album_score(
    release: dict[str, Any],
    target_window: tuple[int, int] | None,
) -> tuple[int, str]:
    """Score a surviving studio album 0-100 like a collector, with reasons."""
    primary_type = str(release.get("primary_type") or release.get("release_group_type") or "").strip().lower()
    release_group_type = str(release.get("release_group_type") or release.get("type") or "").strip().lower()
    year = int(release.get("year") or 0)

    score = 50
    reasons = []

    if primary_type == "album":
        score += 20
        reasons.append("studio album (primary-type album) +20")
    elif release_group_type == "album" or (not primary_type and not release_group_type):
        score += 10
        reasons.append("album +10")

    if target_window and year:
        low, high = target_window
        if low <= year <= high:
            score += 25
            reasons.append(f"{year} within target era {low}-{high} +25")
        elif low - 3 <= year <= high + 3:
            score += 8
            reasons.append(f"{year} near target era {low}-{high} +8")
        else:
            score -= 12
            reasons.append(f"{year} outside target era {low}-{high} -12")

    if year:
        score += 5
        reasons.append("has release year +5")
    else:
        score -= 8
        reasons.append("unknown release year -8")

    score = max(0, min(100, score))
    return score, "; ".join(reasons)


def _rank_artist_albums(
    releases: list[dict[str, Any]],
    target_window: tuple[int, int] | None,
    wants_new: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter to canonical studio albums and score them, best first.

    Returns (all qualifying releases with album_score/score_reason, rejection
    counts). The caller decides how many to take per artist and what to keep in
    reserve for adaptive backfill.
    """
    scored = []
    rejections: dict[str, int] = {}
    for release in releases:
        if not str(release.get("album") or "").strip():
            continue
        reason = _reject_reason(release, target_window, wants_new)
        if reason:
            category = reason.split(":", 1)[0]
            rejections[category] = rejections.get(category, 0) + 1
            continue
        score, score_reason = _album_score(release, target_window)
        scored.append({**release, "album_score": score, "score_reason": score_reason})

    scored.sort(key=lambda item: (
        -int(item.get("album_score") or 0),
        _year_distance(item, target_window),
        (item.get("album") or "").lower(),
    ))
    return scored, rejections


def _global_sort_key(item: dict[str, Any], target_window: tuple[int, int] | None):
    """Best album_score first, then era closeness, LLM confidence, then name."""
    return (
        -int(item.get("album_score") or 0),
        _year_distance(item, target_window),
        -float(item.get("confidence") or 0.0),
        (item.get("artist") or "").lower(),
        (item.get("album") or "").lower(),
    )


def get_real_artist_releases(artist_name: str, prompt: str) -> list[dict[str, Any]]:
    releases = lookup_artist_release_groups(artist_name)
    return [
        {
            "artist": item.get("artist") or artist_name,
            "album": item.get("album") or "",
            "year": int(item.get("year") or 0),
            "musicbrainz_release_group_id": item.get("musicbrainz_release_group_id"),
            "release_group_type": item.get("release_group_type"),
            "primary_type": item.get("primary_type"),
        }
        for item in releases
        if item.get("album")
    ]


def build_missing_albums_request(prompt: str, count: int, format_type: str, library: dict[str, Any]) -> dict[str, Any]:
    summary = build_library_summary(library)
    owned = set()
    for artist in library.get("artists", []):
        artist_name = artist.get("name", "")
        for album in artist.get("albums", []) or []:
            owned.add((artist_name, album))

    owned_artists = [artist.get("name", "") for artist in library.get("artists", []) if artist.get("name")]

    prompt_text = f"""
You are helping the listener discover artists likely missing from their music library.

User request: {prompt}
Requested format: {format_type}

{summary}

Owned artists: {', '.join(owned_artists[:20])}

Return JSON only.
Format:
{{
  "suggestions": [
    {{
      "artist": "Artist Name",
      "reason": "Why it fits the listener's tastes",
      "confidence": 0.9
    }}
  ]
}}

Rules:
- Prefer artists, genres, eras, and moods already reflected in the library.
- Do not suggest artists already in the library summary.
- Return up to {count} suggestions.
- Keep suggestions varied and relevant to the request.
- Do not invent artist names.
"""

    result = providers.ollama_generate(prompt_text, MODEL, timeout=180,
                                       format="json")

    data = json.loads(result["response"])
    suggestions = data.get("suggestions", [])
    llm_artist_selection = len(suggestions)

    normalized = []
    for item in suggestions:
        artist = (item.get("artist") or "").strip()
        if not artist:
            continue
        normalized.append({
            "artist": artist,
            "reason": (item.get("reason") or "Related to your existing library").strip(),
            "confidence": float(item.get("confidence") or 0.5),
        })

    artists_suggested = len(normalized)

    filtered = []
    seen = set()
    for suggestion in normalized:
        artist = suggestion["artist"]
        if artist.lower() in seen:
            continue
        seen.add(artist.lower())
        if any(artist.lower() == owned_artist.lower() for owned_artist in [name for name in [a.get("name", "") for a in library.get("artists", [])] if name]):
            continue
        filtered.append(suggestion)

    owned_artist_filtered = len(filtered)

    target_window = _detect_target_window(prompt)
    wants_new = _wants_new_releases(prompt)

    def as_candidate(suggestion: dict[str, Any], release: dict[str, Any]) -> dict[str, Any]:
        return {
            "artist": suggestion["artist"],
            "album": release["album"],
            "year": release["year"],
            "musicbrainz_release_group_id": release.get("musicbrainz_release_group_id"),
            "reason": suggestion["reason"],
            "confidence": suggestion["confidence"],
            "album_score": release["album_score"],
            "score_reason": release["score_reason"],
        }

    grounded = []       # up to PER_ARTIST_CAP per artist (diversity pass)
    reserve = []        # extra qualifying studio albums, held for backfill
    release_groups_fetched = 0
    rejections: dict[str, int] = {}
    for suggestion in filtered[:count]:
        releases = get_real_artist_releases(suggestion["artist"], prompt)
        release_groups_fetched += len(releases)
        ranked_albums, artist_rejections = _rank_artist_albums(releases, target_window, wants_new)
        for category, amount in artist_rejections.items():
            rejections[category] = rejections.get(category, 0) + amount
        for release in ranked_albums[:PER_ARTIST_CAP]:
            grounded.append(as_candidate(suggestion, release))
        for release in ranked_albums[PER_ARTIST_CAP:]:
            reserve.append(as_candidate(suggestion, release))

    studio_album_candidates = len(grounded) + len(reserve)
    rejected_total = sum(rejections.values())

    # Drop releases already owned (and de-duplicate by artist+album). Reuses the
    # shared helper instead of an inline reimplementation.
    owned_album_filtered = filter_suggestions(grounded, owned)
    owned_album_filtered.sort(key=lambda item: _global_sort_key(item, target_window))

    final_ranked = owned_album_filtered[:count]

    # Adaptive backfill: if the diversity pass came up short, add more qualifying
    # studio albums from already-approved artists (still real, still junk-filtered,
    # still scored) until we reach count or run out of good candidates.
    backfilled_count = 0
    if len(final_ranked) < count and reserve:
        already = {(item["artist"], item["album"]) for item in final_ranked}
        backfill_pool = [
            item for item in filter_suggestions(reserve, owned)
            if (item["artist"], item["album"]) not in already
            and int(item.get("album_score") or 0) >= MIN_BACKFILL_SCORE
        ]
        backfill_pool.sort(key=lambda item: _global_sort_key(item, target_window))
        needed = count - len(final_ranked)
        backfill = backfill_pool[:needed]
        backfilled_count = len(backfill)
        final_ranked = final_ranked + backfill
        final_ranked.sort(key=lambda item: _global_sort_key(item, target_window))

    return {
        "prompt": prompt,
        "count": count,
        "format": format_type,
        "suggestions": final_ranked,
        "pipeline_counts": {
            # Back-compatible keys.
            "llm_artist_selection": llm_artist_selection,
            "artists_suggested": artists_suggested,
            "release_groups_fetched": release_groups_fetched,
            "rejected_by_type_title_year": rejected_total,
            "after_owned_artist_filter": owned_artist_filtered,
            "owned_filtered": len(owned_album_filtered),
            "final_candidates": len(final_ranked),
            "musicbrainz_releases": studio_album_candidates,
            "owned_album_filtering": len(owned_album_filtered),
            "type_filtering": len(owned_album_filtered),
            "final_ranking": len(final_ranked),
            # Album-ranking candidate flow.
            "target_window": list(target_window) if target_window else None,
            "wants_new_releases": wants_new,
            "studio_album_candidates": studio_album_candidates,
            "rejected_total": rejected_total,
            "rejections": rejections,
            "backfilled_count": backfilled_count,
            "backfill_score_floor": MIN_BACKFILL_SCORE,
            "returned": len(final_ranked),
        },
    }


def build_missing_albums_response(prompt: str, count: int, format_type: str, library: dict[str, Any]) -> dict[str, Any]:
    try:
        result = build_missing_albums_request(prompt, count, format_type, library)
    except Exception as exc:
        # Degrade gracefully (e.g. Ollama unreachable or bad JSON) instead of a
        # 500, matching the /station endpoint's behavior.
        logger.exception("missing albums lookup failed")
        return {
            "ok": False,
            "error": "missing albums lookup failed; see server log",
            "prompt": prompt,
            "count": 0,
            "format": format_type,
            "suggestions": [],
            "pipeline_counts": {},
        }
    suggestions = []
    for item in result["suggestions"]:
        artist = item["artist"]
        musicbrainz_release_group_id = item.get("musicbrainz_release_group_id")
        lidarr_search = artist
        if musicbrainz_release_group_id:
            lidarr_search = f"lidarr:{musicbrainz_release_group_id}"
        suggestions.append({
            "artist": artist,
            "album": item.get("album"),
            "year": item.get("year"),
            "musicbrainz_release_group_id": item.get("musicbrainz_release_group_id"),
            "album_score": item.get("album_score"),
            "score_reason": item.get("score_reason"),
            "reason": item["reason"],
            "confidence": item["confidence"],
            "lidarr_search": lidarr_search,
        })
    return {
        "ok": True,
        "prompt": prompt,
        "count": len(suggestions),
        "format": format_type,
        "suggestions": suggestions,
        "pipeline_counts": result.get("pipeline_counts", {}),
    }


def main():
    prompt = " ".join(sys.argv[1:]).strip() or "90s rock essentials"
    with open(INDEX, encoding="utf-8") as handle:
        library = json.load(handle)
    result = build_missing_albums_response(prompt, 10, "lidarr", library)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
