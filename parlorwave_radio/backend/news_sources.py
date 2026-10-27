"""Bulletin SOURCES: the Newsroom's multi-source model.

One station may carry more than one newsroom -- a common shape is
a bulletin at the top of the hour from one publisher and another at
half past from a second. This module owns everything about that list and
nothing about airing it:

  * the stored shape of a source, and its validation
  * migration from the single-feed document that came before
  * which source (if any) is the one this moment is working toward
  * the per-source view the existing fetch path already knows how to read

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not schedule, render,
queue or air anything. news_schedule still owns the one event lifecycle,
because only one bulletin can be on air at a time however many sources are
configured -- audio is serial, and a second scheduler would be a second
set of race conditions against the same queue. This module answers "whose
turn is it", and the proven machine below it does the rest unchanged.

FACTORY STATE IS BLANK, and stays blank. No publisher is shipped
preconfigured, here or anywhere else: an empty source list is the normal
state of a fresh installation, and the Newsroom says so.

IDENTITY IS THE FEED URL, not a position in a list. A source's id is
derived from its canonicalised feed URL by the same function the single
feed always used (news_bulletins.custom_source_id), which is why the
migration below can carry the old last-aired bookkeeping across without
translating anything: the id it lands on is the id that bookkeeping was
already filed under.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any

import news_bulletins

#: A source's schedule is a minute of every hour. Any minute is accepted --
#: the air window's arithmetic does not care which -- but there is
#: deliberately no cron syntax, no "every N minutes" and no per-day rules:
#: an hourly bulletin at a stated minute is the whole model.
MIN_MINUTE, MAX_MINUTE = 0, 59
DEFAULT_TARGET_MINUTE = 30

MAX_SOURCES = 8          # bounded; a station is not a feed reader
MAX_NAME = 60
MAX_URL = 500

#: The keys a stored source document may carry. Closed on purpose: a typo
#: is a refusal while the person is still looking at the box, rather than
#: an inert value that sits in the settings document looking like it works.
SOURCE_KEYS = ("id", "name", "feed_url", "enabled", "target_minute")


class SourceError(ValueError):
    """An invalid source document. The message is safe to show a caller."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")


def source_id_for(feed_url: str, name: str = "") -> str:
    """A source's stable id.

    From the canonicalised feed URL wherever there is one, via the same
    function the single-feed document used -- so a migrated source keeps
    the id its own aired-bulletin history is already filed under, and a
    migration that runs twice lands on the same id both times.

    A source with no URL yet (a half-finished row) still needs to be
    addressable, so it falls back to its name, and finally to a digest of
    nothing in particular -- never to a list index, which changes when a
    row above it is removed.
    """
    url = (feed_url or "").strip()
    if url:
        try:
            return news_bulletins.custom_source_id(url)
        except Exception:  # noqa: BLE001 - an unusable URL is still addressable
            pass
    slug = _slug(name)
    if slug:
        return f"src-{slug[:40]}"
    return "src-" + hashlib.sha256(
        f"{url}|{name}".encode("utf-8")).hexdigest()[:12]


def _text(value: Any, cap: int) -> str:
    return value.strip()[:cap] if isinstance(value, str) else ""


def _coerce_minute(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_TARGET_MINUTE
    return value if MIN_MINUTE <= value <= MAX_MINUTE else DEFAULT_TARGET_MINUTE


def normalize_source(raw: Any, *, index: int = 0) -> dict[str, Any]:
    """One stored source document, read tolerantly.

    Reading is forgiving and WRITING is strict (validate_sources): a
    settings document that somehow holds a bad value must still produce a
    station that plays music, so a malformed field degrades to its default
    here rather than taking the Newsroom down.
    """
    raw = raw if isinstance(raw, dict) else {}
    # A non-string is DROPPED, never stringified: str(5) is "5", which is a
    # feed address that looks configured, fails every fetch, and reads on
    # screen as a typo the owner did not make.
    url = _text(raw.get("feed_url"), MAX_URL)
    name = _text(raw.get("name"), MAX_NAME)
    sid = _text(raw.get("id"), 200) or source_id_for(url, name)
    return {
        "id": sid,
        "name": name,
        "feed_url": url,
        "enabled": bool(raw.get("enabled", True)),
        "target_minute": _coerce_minute(raw.get("target_minute")),
    }


def migrate(bulletins: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The source list for a stored bulletins document, migrating the
    single-feed shape when that is what is there.

    Deterministic, idempotent and non-destructive:

      * a document that already carries `sources` is read as-is and the
        legacy keys beside it are ignored, so running this again changes
        nothing;
      * a document with no `sources` produces ONE source from the old
        `source_name` / `publisher.custom_feed_url` / `schedule.
        target_minute` / `enabled` fields, under the id that document's
        aired-bulletin history is already filed under;
      * a document with neither produces the empty list, which is the
        factory state and not an error.

    Nothing is deleted here: this returns a list, and the caller decides
    what to store. The legacy keys are left in the document on purpose --
    a rollback to the previous build must still find its feed.
    """
    b = bulletins if isinstance(bulletins, dict) else {}
    existing = b.get("sources")
    if isinstance(existing, list) and existing:
        return [normalize_source(s, index=i) for i, s in enumerate(existing)]

    # resolve_publisher_source is the seam that already understood BOTH
    # legacy shapes -- a typed custom URL and a built-in registry id -- and
    # it returns the very id the cached bulletins and the aired-token ring
    # are already filed under. Migrating through it is what makes the old
    # last-aired state carry across with nothing translated.
    try:
        sid, url, label = news_bulletins.resolve_publisher_source(b)
    except Exception:                  # noqa: BLE001 - unusable / unconfigured
        return []                      # never configured: the factory state
    if not url:
        return []
    name = str(b.get("source_name") or "").strip() or str(label or "").strip()
    minute = ((b.get("schedule") or {}).get("target_minute"))
    return [normalize_source({
        "id": sid,
        "name": name,
        "feed_url": url,
        # The legacy document's single `enabled` was the whole Newsroom's
        # gate and remains so; the migrated source is enabled in its own
        # right, or an owner who turns the Newsroom back on would find its
        # only source switched off for no reason they ever chose.
        "enabled": True,
        "target_minute": minute,
    })]


def sources_of(bulletins: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every configured source, migrating on read. The one accessor: no
    caller reads `bulletins["sources"]` directly, so the legacy shape is
    understood everywhere or nowhere."""
    return migrate(bulletins)


def enabled_sources(bulletins: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Sources that may actually air: switched on, and with a feed to
    fetch. A source with no address is a half-finished row, not a fault."""
    return [s for s in sources_of(bulletins)
            if s["enabled"] and s["feed_url"]]


def by_id(bulletins: dict[str, Any] | None, source_id: str | None
          ) -> dict[str, Any] | None:
    for s in sources_of(bulletins):
        if s["id"] == str(source_id or ""):
            return s
    return None


def validate_sources(raw: Any) -> None:
    """Strict validation for a WRITE. Raises SourceError with a message a
    person can act on."""
    if not isinstance(raw, list):
        raise SourceError("bulletins.sources must be a list")
    if len(raw) > MAX_SOURCES:
        raise SourceError(f"at most {MAX_SOURCES} bulletin sources")
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for i, item in enumerate(raw):
        where = f"bulletins.sources[{i}]"
        if not isinstance(item, dict):
            raise SourceError(f"{where} must be an object")
        extra = set(item) - set(SOURCE_KEYS)
        if extra:
            raise SourceError(
                f"unknown {where} fields: {', '.join(sorted(extra))}")
        name = item.get("name", "")
        if not isinstance(name, str) or len(name) > MAX_NAME:
            raise SourceError(
                f"{where}.name must be a string of at most {MAX_NAME} characters")
        if "\n" in name or "\r" in name:
            raise SourceError(f"{where}.name must be a single line")
        if not isinstance(item.get("enabled", True), bool):
            raise SourceError(f"{where}.enabled must be a boolean")
        minute = item.get("target_minute", DEFAULT_TARGET_MINUTE)
        if (isinstance(minute, bool) or not isinstance(minute, int)
                or not (MIN_MINUTE <= minute <= MAX_MINUTE)):
            raise SourceError(
                f"{where}.target_minute must be a whole number of minutes "
                f"from {MIN_MINUTE} to {MAX_MINUTE}")
        url = item.get("feed_url", "")
        if not isinstance(url, str) or len(url) > MAX_URL:
            raise SourceError(
                f"{where}.feed_url must be a string of at most {MAX_URL} characters")
        if url.strip():
            _validate_url(url.strip(), where)
            canon = news_bulletins.canonicalize_feed_url(url.strip())
            if canon in seen_urls:
                raise SourceError(
                    "two bulletin sources cannot share one feed address")
            seen_urls.add(canon)
        sid = item.get("id")
        if sid is not None:
            if not isinstance(sid, str) or not sid.strip():
                raise SourceError(f"{where}.id must be a non-empty string")
            if sid in seen_ids:
                raise SourceError("two bulletin sources cannot share one id")
            seen_ids.add(sid)


def _validate_url(url: str, where: str) -> None:
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise SourceError(f"{where}.feed_url must be an http(s) URL")
    if not parts.hostname:
        raise SourceError(f"{where}.feed_url must name a host")
    if "@" in parts.netloc:
        raise SourceError(
            f"{where}.feed_url must not carry credentials in the URL")


# --- whose turn is it ---------------------------------------------------

def due_source(bulletins: dict[str, Any] | None, local_now: datetime,
               active_target_fn) -> tuple[dict[str, Any] | None, datetime | None]:
    """The source this moment is working toward, and its target time.

    Every enabled source has its own minute of the hour, so each has its
    own next target; the soonest one is the one being worked toward. Ties
    are broken by the order the sources are configured in -- a stable,
    visible rule rather than a dictionary's iteration order -- so two
    sources that come due for the SAME boundary always resolve the same
    way, and the loser is skipped for that hour rather than racing.

    `active_target_fn(local_now, minute)` is news_schedule's own function,
    passed in rather than imported: this module must not depend on the
    scheduler it is consulted by.

    (None, None) when nothing is configured, which is the factory state.
    """
    best: tuple[datetime, int, dict[str, Any]] | None = None
    for order, src in enumerate(enabled_sources(bulletins)):
        target = active_target_fn(local_now, src["target_minute"])
        key = (target, order)
        if best is None or key < (best[0], best[1]):
            best = (target, order, src)
    if best is None:
        return None, None
    return best[2], best[0]


def contenders_for(bulletins: dict[str, Any] | None, target_at: datetime,
                   active_target_fn) -> list[dict[str, Any]]:
    """Every enabled source whose own target is this exact moment.

    Normally one. More than one means two newsrooms were scheduled for the
    same minute; they are returned in configured order, which is the order
    due_source resolves them in, so a caller can say plainly which one took
    the boundary and which one is waiting for the next hour.
    """
    out = []
    for src in enabled_sources(bulletins):
        if active_target_fn(target_at, src["target_minute"]) == target_at:
            out.append(src)
    return out


# --- the view the existing fetch path already understands ---------------

def as_feed_config(source: dict[str, Any] | None,
                   bulletins: dict[str, Any] | None = None) -> dict[str, Any]:
    """ONE source, shaped like the single-feed document that came before.

    news_bulletins.resolve_publisher_source and fetch_latest_bulletin read
    `source_name` and `publisher.custom_feed_url`, and have been fetching,
    validating and storing bulletins correctly for a long time. Rather than
    teach them about a list -- and re-earn that confidence -- each source is
    handed to them in the shape they already know. The multi-source model
    therefore adds no new download, parse, duration or storage path.
    """
    src = source or {}
    mode = ((bulletins or {}).get("mode")
            or news_bulletins.ORIGIN_PUBLISHER)
    sid = str(src.get("id") or "")
    # A source migrated from a built-in registry entry keeps that entry's
    # id, and resolve_publisher_source must land on the SAME id it always
    # did -- the cached bulletin rows and the aired-token ring are filed
    # under it. Re-describing it as a custom URL would silently change its
    # identity and make every cached bulletin look like another source's.
    if sid and sid in getattr(news_bulletins, "BUILTIN_SOURCES", {}):
        publisher = {"source": sid, "custom_feed_url": ""}
    else:
        publisher = {"source": "custom",
                     "custom_feed_url": src.get("feed_url") or ""}
    return {
        "enabled": bool(src.get("enabled", True)),
        "mode": mode,
        "source_name": src.get("name") or "",
        "schedule": {"target_minute": _coerce_minute(src.get("target_minute"))},
        "publisher": publisher,
    }


def display_name(source: dict[str, Any] | None) -> str:
    """What the station calls this source on air and on screen."""
    src = source or {}
    return news_bulletins.configured_source_name(
        {"source_name": src.get("name") or ""}, src.get("feed_url") or "")
