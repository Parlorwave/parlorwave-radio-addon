#!/usr/bin/env python3
"""Shared MusicBrainz release-group classification.

Decides whether a MusicBrainz release group is a canonical studio album,
independent of any era/prompt preferences. Both the missing-albums recommender
and Collector Intelligence use this so there is a single definition of "studio
album". Year/recency filtering (e.g. rejecting 2020+ for a "90s" request) is the
caller's concern, not this module's.
"""

import re
from typing import Any

# Release-group types that are not standard studio albums.
JUNK_TYPES = {
    "live", "compilation", "ep", "single", "broadcast", "demo",
    "interview", "remix", "collection", "box set", "box-set", "video",
}

# Title tokens split by confidence (matched on word boundaries so "live" no
# longer matches "Alive"/"deliverance", etc.). STRONG tokens are compilation/
# edition phrases that reject regardless of MB type. WEAK tokens are ambiguous
# single words that reject only when the MB type is unresolved -- a release MB
# has resolved to "Album" is trusted over a weak title heuristic. This keeps
# real studio albums like "Live Through This" and "Deluxe" from being rejected.
STRONG_JUNK_TITLE_TOKENS = [
    "greatest hits", "best of", "complete studio albums",
    "deluxe edition", "expanded edition", "anniversary",
    "box set", "box-set",
]
WEAK_JUNK_TITLE_TOKENS = [
    "deluxe", "live", "compilation", "remix",
    "interview", "broadcast", "demo", "collection",
]
# Back-compat combined list (title_has_junk_token uses it).
JUNK_TITLE_TOKENS = STRONG_JUNK_TITLE_TOKENS + WEAK_JUNK_TITLE_TOKENS


def _token_re(tokens: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b")


_STRONG_TITLE_RE = _token_re(STRONG_JUNK_TITLE_TOKENS)
_WEAK_TITLE_RE = _token_re(WEAK_JUNK_TITLE_TOKENS)
_JUNK_TITLE_RE = _token_re(JUNK_TITLE_TOKENS)


def normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


_EDITION_SUFFIX_RE = re.compile(
    r"[(\[][^)\]]*\b(deluxe|remaster(?:ed)?|edition|anniversary|bonus"
    r"|expanded|reissue|special|legacy)\b[^)\]]*[)\]]"
)
_PUNCT_RE = re.compile(r"[^\w\s]")


def match_key(title: str) -> str:
    """Aggressive key for owned-vs-MB album matching: lowercase, drop edition
    parentheticals ("(Deluxe Edition)", "[Remastered]"), strip punctuation,
    collapse whitespace. NOT for display or junk detection."""
    text = normalize_title(title)
    text = _EDITION_SUFFIX_RE.sub(" ", text)
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())


def title_has_junk_token(normalized_title: str) -> bool:
    return bool(_JUNK_TITLE_RE.search(normalized_title))


def studio_album_reject_reason(release: dict[str, Any]) -> str | None:
    """Return why a release is not a studio album, or None if it is one.

    Categories: ``type`` (live/comp/EP/single/etc.), ``title`` (deluxe/
    anniversary/greatest hits/...) and ``non_album`` (unresolved type). This is
    era-agnostic: a 2021 studio album is still a studio album here.

    Title handling is confidence-tiered: STRONG title phrases (greatest hits,
    deluxe edition, ...) always reject; a release MB has resolved to ``Album`` is
    then trusted; only for unresolved types do WEAK single-word tokens (live,
    deluxe, ...) reject. So "Live Through This"/"Deluxe" tagged as Album survive,
    while "Live at Budokan" with no type still rejects.
    """
    primary_type = str(release.get("primary_type") or release.get("release_group_type") or "").strip().lower()
    release_group_type = str(release.get("release_group_type") or release.get("type") or "").strip().lower()
    normalized_title = normalize_title(release.get("album") or "")

    if primary_type in JUNK_TYPES:
        return f"type:{primary_type}"
    if release_group_type in JUNK_TYPES:
        return f"type:{release_group_type}"
    if _STRONG_TITLE_RE.search(normalized_title):
        return "title"
    if primary_type == "album" or release_group_type == "album":
        return None
    if _WEAK_TITLE_RE.search(normalized_title):
        return "title"
    return "non_album"


def is_studio_album(release: dict[str, Any]) -> bool:
    """True if the release group is a canonical studio album (era-agnostic)."""
    return studio_album_reject_reason(release) is None
