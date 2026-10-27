"""News bulletin model: source identity, feed parsing, and the
news_bulletins artifact rows (sessions.db schema v8).

A news_bulletin row is the final playable bulletin ARTIFACT. origin_type
records how it was produced -- publisher_audio (RSS enclosure download)
today, generated_briefing (News Board -> grounded LLM script -> TTS) as an
accepted future origin -- and everything at or after the 'cached' state is
origin-agnostic: serving, preview, retention and eventual queue insertion
read only token/state/audio_path, never the origin.

Scope: registry, canonicalization, identity, pure parser,
newest-viable selection, row helpers, and a bounded fetch/download/validate
pipeline behind ONE global lock. Still absent by design: any scheduler, any
Music Assistant call, any automatic trigger. Nothing here runs unless a
caller explicitly invokes it -- app startup, settings load, DJ rendering
and the reaper all leave this module alone -- and the only HTTP surface is
a passive tokenized audio route.

AUDIO IS ONE SLOT PER SOURCE. ParlorWave is not a podcast archive: exactly one
validated bulletin MP3 is retained FOR EACH CONFIGURED SOURCE
(news-<token>.mp3), a replacement stages beside it (.news-<token>.tmp) and
the old file is deleted only after the database switch commits. Metadata
rows stay (bounded) so old tokens can answer 410.

WHY PER SOURCE AND NOT ONE GLOBALLY. The slot used to be global, and with
two newsrooms configured that made the two of them fight over it: whichever
fetched last held the only cached row, so the other source's screen row
went blank, its Listen Again disappeared the moment its neighbour refreshed,
and the planner had to re-download a bulletin it had downloaded an hour
earlier. The rule a station actually wants is a simpler one --
the latest bulletin of EACH newsroom stays playable, and nothing older than
that does. Storage is still bounded, by the number of configured sources
(MAX_SOURCES) rather than by one.

States (v0): discovered -> downloading -> cached | failed, plus expired
(retention terminal). Future origins/states are additive TEXT values.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import secrets as _secrets
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import service_clock

logger = logging.getLogger("musicbrain")

ORIGIN_PUBLISHER = "publisher_audio"
ORIGIN_GENERATED = "generated_briefing"
# Which of those modes this build can actually produce. settings validation
# accepts both origins because the SHAPE is valid either way; this is the
# separate question of what the system can serve today, and it is the one
# the settings write route checks so a mode that cannot work is refused at
# the door instead of silently persisting and failing later. A future
# generated-briefing commit adds ORIGIN_GENERATED here and nowhere else.
IMPLEMENTED_MODES = (ORIGIN_PUBLISHER,)

# Publisher v0 is MP3-only: the whole finite-audio path (serving, MA's own
# probing, duration validation) is proven against MPEG audio. Non-MP3
# enclosures yield an explicit unsupported outcome, never a silent pass.
#
# ONE FORMAT, SEVERAL SPELLINGS. RFC 3003 registers `audio/mpeg` and that
# is what a correct server sends for an MP3. The others below are not
# registered and real publishers send them anyway, for byte-identical
# files -- refusing them would reject a perfectly ordinary bulletin with
# "unsupported content type: audio/mp3" and leave that source unplayable
# for good. Accepting them is not a widening of what this build can play: it is
# the same one format, and the gate that actually decides is
# `_validate_audio`, which reads the signature and probes the duration. A
# HINT MAY NOT BE THE THING THAT REFUSES -- a server that calls a web page
# an MP3 still gets refused, on the bytes, one step later.
MP3_CONTENT_TYPE = "audio/mpeg"
SUPPORTED_AUDIO_TYPES = (MP3_CONTENT_TYPE, "audio/mp3", "audio/x-mp3",
                         "audio/mpeg3", "audio/x-mpeg-3")

# The registry of built-in publisher sources, and it is EMPTY in the public
# product -- deliberately, and this is the whole of the decision.
#
# A built-in entry would be a named news organisation and its podcast
# feed URL. That is somebody else's newsroom, and shipping it would
# mean every installation arriving with a publisher already chosen for it,
# that publisher's brand in the public status surface and in the settings
# vocabulary, and a distribution relationship nobody has. The Newsroom asks
# for a feed and says so until it has one; a customer supplies whichever
# audio podcast or news RSS feed they want, and `custom` is how every
# public installation reaches the bulletin path.
#
# The MECHANISM survives, and is exercised by the tests, because a future
# preset that is genuinely ours (a Parlorwave-produced briefing, say) slots
# in here without a schema change. Registry ids never contain ':' -- the
# source_id column namespaces customs as 'custom:<hash>' and future
# briefing profiles as 'briefing:<profile_id>', so the three kinds can
# never collide.
#
# An installation that stored a retired id keeps it in its settings
# document (load is lenient by design) and resolves to "no usable source",
# which is what the Newsroom already knows how to say. Nothing is migrated
# and nothing is guessed.
BUILTIN_SOURCES: dict[str, dict[str, str]] = {}


class FeedError(ValueError):
    """Invalid feed content or unusable source configuration. Message is
    safe to show in the developer UI."""


class PayloadTooLarge(FeedError):
    """The server answered normally; the body exceeded its cap.

    A FeedError so every existing handler keeps treating it exactly as it
    did, and a distinct type so the one caller that reports a REASON to a
    person can stop calling this "unreachable". The host resolved, the
    connection opened and bytes arrived -- telling the owner to check the
    address sends them to look at the one thing that was not wrong."""


# --- source identity ----------------------------------------------------------

def canonicalize_feed_url(url: str) -> str:
    """Deterministic LOCAL identity form of a custom feed URL.

    Lowercases scheme and host, drops the scheme-default port and any
    fragment, and normalizes an empty path to '/'. Path and query are
    preserved byte-for-byte (they are case-sensitive on real servers).
    Deliberately NO redirect resolution, DNS, percent-encoding rewrites or
    IDN folding -- this is identity, not equivalence detection."""
    parts = urlsplit((url or "").strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port is not None and not ((scheme == "http" and port == 80)
                                 or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunsplit((scheme, host, parts.path or "/", parts.query, ""))


def custom_source_id(url: str) -> str:
    """Stable internal identity for a custom feed: 'custom:' + the first 16
    hex chars of sha256 over the canonical URL. Identity, not a secret --
    the serving credential is the separate random row token."""
    canon = canonicalize_feed_url(url)
    return "custom:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def source_name_from_feed(feed_url: str) -> str:
    """An honest name for a feed nobody has named: its host, minus 'www.'.

    A hostname is a FACT about the configured feed. It is not a claim about
    who produced the audio, which is why it is only ever the fallback for a
    name the station owner did not supply."""
    host = (urlsplit(feed_url or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def configured_source_name(bulletins: dict[str, Any] | None,
                           feed_url: str = "") -> str:
    """What the station calls its news source, in the owner's words.

    The owner's `source_name` wins; otherwise the feed host; otherwise the
    neutral fallback. Never 'custom:<hash>' (internal identity) and never a
    publisher this build merely happens to know about."""
    name = ((bulletins or {}).get("source_name") or "").strip()
    return name or source_name_from_feed(feed_url) or "Newsroom"


def resolve_publisher_source(bulletins: dict[str, Any]) -> tuple[str, str, str]:
    """(source_id, feed_url, label) for the configured publisher source.
    Built-in ids pass through verbatim; 'custom' resolves through
    custom_source_id. Raises FeedError on unusable configuration.

    The label is PRESENTATION: it is what the intro speaks and what the
    queue shows, so a name the station owner typed outranks anything this
    build could infer -- including a built-in's own curated label."""
    pub = (bulletins or {}).get("publisher") or {}
    src = pub.get("source") or ""
    owner_name = ((bulletins or {}).get("source_name") or "").strip()
    if src == "custom":
        url = (pub.get("custom_feed_url") or "").strip()
        if not url:
            raise FeedError("custom feed URL is not configured")
        return custom_source_id(url), url, configured_source_name(bulletins, url)
    entry = BUILTIN_SOURCES.get(src)
    if entry is None:
        raise FeedError(f"unknown built-in source: {src}")
    return src, entry["feed_url"], owner_name or entry["label"]


# --- pure feed parser (no I/O) ------------------------------------------------

def _parse_pubdate(text: str | None) -> str | None:
    """RFC-822 pubDate -> aware ISO string, or None when malformed. Naive
    results (e.g. '-0000') are pinned to UTC so newest-selection can compare
    every entry without aware/naive crashes."""
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text.strip())
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _parse_duration(text: str | None) -> float | None:
    """'280' / '4:40' / '0:04:40' -> seconds, else None."""
    if not text:
        return None
    parts = text.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if not nums or len(nums) > 3:
        return None
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return float(seconds)


def parse_feed(xml_bytes: bytes, feed_url: str) -> list[dict[str, Any]]:
    """Parse RSS/podcast XML into entry dicts. PURE: no network, no DB.

    Per item: publisher <guid> is the preferred identity; when absent the
    enclosure URL substitutes (flagged via guid_from_enclosure). Duplicate
    identities within one feed keep the first occurrence. The first
    supported (MP3, however the feed spells it) enclosure wins; an item
    without one is KEPT with
    enclosure_url=None so selection can report unsupported_format honestly
    instead of silently skipping. Items with neither guid nor enclosure are
    unidentifiable and dropped. Malformed pubDate keeps the entry with
    published_at=None. Empty feed -> []. Malformed XML -> FeedError."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FeedError(f"malformed feed XML: {exc}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in root.iter("item"):
        enc_url = enc_type = None
        expected_length = None
        for enc in item.findall("enclosure"):
            ctype = (enc.get("type") or "").split(";")[0].strip().lower()
            if ctype in SUPPORTED_AUDIO_TYPES and enc.get("url"):
                enc_url = enc.get("url")
                enc_type = ctype
                raw_len = (enc.get("length") or "").strip()
                expected_length = int(raw_len) if raw_len.isdigit() else None
                break
        guid = (item.findtext("guid") or "").strip()
        guid_from_enclosure = False
        if not guid:
            if not enc_url:
                continue                      # no identity at all: dropped
            guid = enc_url
            guid_from_enclosure = True
        if guid in seen:
            continue                          # first occurrence wins
        seen.add(guid)
        duration = None
        for child in item:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "duration":
                duration = _parse_duration(child.text)
                break
        entries.append({
            "source_guid": guid,
            "guid_from_enclosure": guid_from_enclosure,
            "title": (item.findtext("title") or "").strip(),
            "published_at": _parse_pubdate(item.findtext("pubDate")),
            "enclosure_url": enc_url,
            "content_type": enc_type,
            "expected_length": expected_length,
            "duration_s": duration,
        })
    return entries


def select_newest_viable(entries: list[dict[str, Any]],
                         ) -> tuple[dict[str, Any] | None, str | None]:
    """(entry, None) for the newest usable publisher entry, else
    (None, reason). Newest = max published_at (aware-normalized ISO,
    compared as datetimes); entries without a parseable date fall back to
    document order (first wins) rather than trusting any feed to be
    newest-first. If the NEWEST item has no supported MP3 enclosure the
    result is an explicit ('unsupported_format') -- never a silent
    substitution of an older item."""
    if not entries:
        return None, "no_items"
    dated = [e for e in entries if e.get("published_at")]
    if dated:
        newest = max(dated,
                     key=lambda e: datetime.fromisoformat(e["published_at"]))
    else:
        newest = entries[0]
    if not newest.get("enclosure_url"):
        return None, "unsupported_format"
    declared = newest.get("duration_s")
    if declared is not None and declared > MAX_BULLETIN_S:
        # Refused on the feed's own word, before a byte is downloaded. Like
        # unsupported_format this is explicit about the NEWEST item and
        # never a silent substitution of an older one: a station that
        # quietly aired yesterday's news because today's ran long would be
        # harder to diagnose than one that says why it aired nothing.
        return None, TOO_LONG_REASON
    return newest, None


# --- artifact rows (sessions.db, schema v8) -----------------------------------

_COLS = ("token", "origin_type", "source_id", "source_guid", "title",
         "published_at", "feed_url", "enclosure_url", "content_type",
         "expected_length", "audio_path", "size_bytes", "duration_s",
         "produced_at", "state", "error", "created_at", "updated_at")


def _iso(now: datetime | None = None) -> str:
    return (now or service_clock.legacy_now()).isoformat(timespec="seconds")


def _connect(db_path=None):
    import playback
    playback.ensure_sessions_schema(db_path)
    return playback._connect(db_path)


def _row_dict(row) -> dict[str, Any]:
    return dict(zip(_COLS, row))


def get_bulletin(token: str, db_path=None) -> dict[str, Any] | None:
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_COLS)} FROM news_bulletins WHERE token = ?",
            (token,)).fetchone()
    finally:
        c.close()
    return _row_dict(row) if row else None


def upsert_publisher_entry(source_id: str, feed_url: str,
                           entry: dict[str, Any],
                           now: datetime | None = None,
                           db_path=None) -> dict[str, Any]:
    """Idempotent discovery of one publisher entry. Publisher identity is
    UNIQUE(source_id, source_guid): the same GUID from the same source is
    one row forever (the fresh token minted here is simply discarded on
    conflict); the same GUID from a DIFFERENT source is a distinct
    bulletin. Returns the surviving row."""
    now_iso = _iso(now)
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO news_bulletins (token, origin_type, source_id, "
            "source_guid, title, published_at, feed_url, enclosure_url, "
            "content_type, expected_length, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id, source_guid) DO NOTHING",
            (_secrets.token_hex(16), ORIGIN_PUBLISHER, source_id,
             entry["source_guid"], entry.get("title") or "",
             entry.get("published_at"), feed_url,
             entry.get("enclosure_url"), entry.get("content_type") or "",
             entry.get("expected_length"), now_iso, now_iso))
        c.commit()
        row = c.execute(
            f"SELECT {', '.join(_COLS)} FROM news_bulletins "
            "WHERE source_id = ? AND source_guid = ?",
            (source_id, entry["source_guid"])).fetchone()
    finally:
        c.close()
    return _row_dict(row)


def create_generated_row(source_id: str, title: str,
                         now: datetime | None = None,
                         db_path=None) -> dict[str, Any]:
    """A generated-briefing artifact row: no publisher GUID, no enclosure.
    Structural support only in v0 -- the generation pipeline (News Board,
    ranking, script, TTS) does not exist yet; future code produces audio
    and hands it to _finalize_artifact like any other origin."""
    now_iso = _iso(now)
    token = _secrets.token_hex(16)
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO news_bulletins (token, origin_type, source_id, "
            "title, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (token, ORIGIN_GENERATED, source_id, title, now_iso, now_iso))
        c.commit()
    finally:
        c.close()
    return get_bulletin(token, db_path)


def _finalize_artifact(token: str, audio_path: str, size_bytes: int,
                       duration_s: float, now: datetime | None = None,
                       db_path=None) -> bool:
    """The origin-agnostic convergence seam: every producer (publisher
    download now, briefing generation later) ends by recording its finished
    audio here. Conditional transition discovered/downloading -> cached;
    exactly one caller can win. Records metadata ONLY -- file placement is
    the producer's job, and nothing here touches the filesystem."""
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE news_bulletins SET state = 'cached', audio_path = ?, "
            "size_bytes = ?, duration_s = ?, produced_at = ?, updated_at = ? "
            "WHERE token = ? AND state IN ('discovered', 'downloading')",
            (audio_path, size_bytes, duration_s, _iso(now), _iso(now), token))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


# --- bounded retrieval ---------------------------------------------------------
#
# Every network bound is enforced on the STREAM, never on a header a server
# might omit or lie about. Custom feeds are configured by the operator, not
# attacker-supplied, so the boundary pinned here is the minimum that keeps a
# hostile RESPONSE cheap: http(s) only, bounded redirects with the scheme
# re-checked at each hop, finite timeouts, hard byte caps. Private-address
# and DNS-rebinding defence is not implemented yet, and must be before any
# public fetch endpoint exists; today there are only the operator's manual
# actions.

# Sized for real news podcast feeds, not guessed: a feed can run to several
# MiB, because a podcast feed carries its whole back catalogue rather than
# today's episodes. A 1 MiB bound would refuse those, so a station could not
# carry a large publisher at all, and the refusal would surface as
# "unreachable" about an address that had answered perfectly. The bound still exists to keep a
# hostile response cheap; it is just set above what real newsrooms publish
# rather than below it.
FEED_MAX_BYTES = 8 * 1024 * 1024          # RSS; large podcast feeds run to several MiB
# Bounded so that the byte cap cannot silently contradict the duration
# policy below. Thirty minutes of spoken-word MP3 is ~14 MB at 64 kbps and
# ~29 MB at 128 kbps; a 25 MiB cap would truncate a legitimate 30-minute
# bulletin at the higher bitrate and report it as a broken download.
AUDIO_MAX_BYTES = 40 * 1024 * 1024        # bulletins are typically several MB
FEED_TIMEOUT_S = float(os.environ.get("NEWS_FEED_TIMEOUT_S", "30"))
AUDIO_TIMEOUT_S = float(os.environ.get("NEWS_AUDIO_TIMEOUT_S", "120"))
MAX_REDIRECTS = 5                         # a few hops are normal (tracker -> CDN)
CHUNK_BYTES = 64 * 1024
MIN_DURATION_S = 30.0

# THE BULLETIN INSERTION MAXIMUM. Thirty minutes, inclusive: 29:59 and
# 30:00 are bulletins, 30:01 is not. This bounds NEWSROOM BULLETIN MODE
# only -- what may be dropped between two songs on a music station without
# the station stopping being one. It is not a statement about long-form
# programming, which is a different feature and will need a different
# limit; the name says which of the two this is so a later feature does
# not inherit this number by accident.
MAX_BULLETIN_S = 1800.0
# Advisory only. Fifteen minutes was the old hard cap; a bulletin that long
# is unusual enough to be worth saying out loud, and not wrong enough to
# refuse.
LONG_BULLETIN_WARN_S = 900.0
#: One sentence, used wherever an over-long episode is refused -- the
#: pre-flight check on the feed's own declaration and the post-download
#: probe alike -- so a listener reading the Newsroom sees one reason and
#: not two spellings of it.
TOO_LONG_ERROR = "Episode too long for a bulletin (%s)"
TOO_LONG_REASON = "episode_too_long"


def _clock_words(seconds: float | None) -> str:
    """M:SS, the way a running time is written. Shown beside the refusal so
    a listener can see WHICH episode was too long without opening the feed."""
    if not seconds or seconds <= 0:
        return "unknown length"
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"
ROW_CAP = 50                              # bounded metadata history
TMP_GRACE_S = 900.0                       # don't reap a live download's temp
USER_AGENT = "ParlorwaveRadio/1.0 (news bulletins)"


class _BoundedRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are bounded AND re-validated: a 302 to file:// or ftp://
    must not become a local read just because the first URL looked fine."""
    max_redirections = MAX_REDIRECTS
    max_repeats = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urlsplit(newurl).scheme.lower()
        if scheme not in ("http", "https"):
            raise FeedError("refusing redirect to unsupported scheme: "
                            + scheme)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_open(url: str, timeout: float):
    """The ONE network seam for news (tests patch this). Bounded, scheme
    checked before the first byte and again at every redirect."""
    scheme = urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise FeedError("unsupported URL scheme: " + (scheme or "(none)"))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_BoundedRedirect)
    return opener.open(req, timeout=timeout)


def _stream_capped(resp, max_bytes: int, write: Callable[[bytes], Any]) -> int:
    """Copy a response body under a hard cap. Content-Length is used as an
    early refusal when present and honest, but the running total is the
    authority -- a missing or lying header changes nothing."""
    declared = (resp.headers.get("Content-Length") or "").strip()
    if declared.isdigit() and int(declared) > max_bytes:
        raise PayloadTooLarge("declared size %s exceeds %d bytes"
                              % (declared, max_bytes))
    total = 0
    while True:
        chunk = resp.read(CHUNK_BYTES)
        if not chunk:
            return total
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLarge("payload exceeded %d bytes" % max_bytes)
        write(chunk)


def fetch_feed_bytes(feed_url: str, open_fn=None) -> bytes:
    """Bounded RSS retrieval. Raises FeedError on anything unusable; the
    caller's audio cache is never touched by a feed failure."""
    buf = io.BytesIO()
    with (open_fn or _http_open)(feed_url, FEED_TIMEOUT_S) as resp:
        _stream_capped(resp, FEED_MAX_BYTES, buf.write)
    return buf.getvalue()


# --- one-slot artifact storage -------------------------------------------------

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")


def news_audio_dir() -> Path:
    """Created on demand by a download, never by import or startup."""
    import paths
    return paths.BASE_DIR / "news_audio"


def _audio_path(token: str) -> Path:
    """Per-token filename, deliberately not a fixed news.mp3: a crash
    between the rename and the database switch must never leave an OLD
    token pointing at NEW bytes."""
    return news_audio_dir() / ("news-" + token + ".mp3")


def _tmp_path(token: str) -> Path:
    return news_audio_dir() / (".news-" + token + ".tmp")


def _probe_duration(path: Path) -> float | None:
    import audio_probe
    return audio_probe.duration(path)


def _validate_audio(path: Path) -> float:
    """Signature + probe gate. Content-Type and URL suffix are hints; this
    is the authority. Returns the duration in seconds."""
    size = path.stat().st_size if path.exists() else 0
    if size < 1024:
        raise FeedError("downloaded audio is empty or implausibly small")
    with path.open("rb") as f:
        head = f.read(3)
    if not (head == b"ID3" or (len(head) > 1 and head[0] == 0xFF
                               and (head[1] & 0xE0) == 0xE0)):
        raise FeedError("downloaded payload is not MP3 audio")
    duration = _probe_duration(path)
    if duration is None:
        raise FeedError("downloaded audio is not a probeable MP3")
    if duration < MIN_DURATION_S:
        raise FeedError("implausible bulletin duration: %.0fs" % duration)
    if duration > MAX_BULLETIN_S:
        # The probe is the authority: a feed may declare any duration it
        # likes, or none. An episode that only reveals its true length
        # after download is still refused, and refused in the same words.
        raise FeedError(TOO_LONG_ERROR % _clock_words(duration))
    return duration


def _download_audio(url: str, tmp: Path, open_fn=None,
                    tag: bytes = b"") -> int:
    """Stream one enclosure into the temp slot. Never buffers the whole
    episode in RAM; any failure removes the temp file.

    `tag` is this station's ID3v2 block, written FIRST; the publisher's
    own leading ID3v2 block, if there is one, is dropped from the stream
    so the artifact carries exactly one tag. The audio frames after it are
    written exactly as received. This is the tagging step: it used to run
    after validation as a read-modify-rewrite of the whole file, so every
    bulletin cost two full writes of its audio (a storage endurance
    cost). Returns the number of bytes received from the network
    (the cap applies to those, as before)."""
    import audio_probe
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (open_fn or _http_open)(url, AUDIO_TIMEOUT_S) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0]
            ctype = ctype.strip().lower()
            if ctype and ctype not in SUPPORTED_AUDIO_TYPES:
                raise FeedError("unsupported content type: " + ctype)
            with tmp.open("wb") as f:
                if tag:
                    f.write(tag)
                    sink = audio_probe.TagStripper(f.write)
                else:
                    sink = f.write
                size = _stream_capped(resp, AUDIO_MAX_BYTES, sink)
                if tag:
                    sink.flush()
                f.flush()
                os.fsync(f.fileno())
        return size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def activate_artifact(token: str, tmp: Path, size_bytes: int,
                      duration_s: float, now: datetime | None = None,
                      db_path=None) -> dict[str, Any]:
    """The origin-agnostic one-slot switch. Any producer (publisher
    download today, briefing generation later) hands over a VALIDATED temp
    file and this performs the crash-safe ordering:

        1. atomically rename temp -> news-<token>.mp3  (old still current)
        2. ONE transaction: this row becomes cached, and every other cached
           row OF THE SAME SOURCE becomes expired
        3. only then delete the superseded audio files

    A crash between 1 and 2 leaves an orphan file the old token can never
    reach; a crash between 2 and 3 leaves a stale file no live token can
    reach. At no point does an old token resolve to new bytes.

    THE SLOT IS THE SOURCE'S, not the station's. Another newsroom's latest
    bulletin is not superseded by this one -- they are different programmes
    that happen to share a disk -- so its row and its file are left exactly
    as they are. Only this source's own previous bulletin is reclaimed,
    which is the rule that makes "the latest bulletin of each source is the
    one you can listen to again" true by construction rather than by
    whichever source refreshed most recently."""
    now_dt = now or service_clock.legacy_now()
    final = _audio_path(token)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)                       # step 1: atomic publish
    c = _connect(db_path)
    superseded = []
    try:
        c.execute("BEGIN IMMEDIATE")             # step 2: one switch
        cur = c.execute(
            "UPDATE news_bulletins SET state = 'cached', audio_path = ?, "
            "size_bytes = ?, duration_s = ?, produced_at = ?, updated_at = ? "
            "WHERE token = ? AND state IN ('discovered', 'downloading')",
            (str(final), size_bytes, duration_s, _iso(now_dt), _iso(now_dt),
             token))
        if cur.rowcount != 1:
            c.rollback()
            final.unlink(missing_ok=True)        # never activated: not ours
            return {"ok": False, "reason": "claim_lost"}
        # WHOSE slot this is, read inside the same transaction rather than
        # taken from a caller: the row is the authority on its own source,
        # and `IS` rather than `=` so a row with no source_id supersedes
        # only other rows with no source_id instead of matching nothing.
        owner = c.execute("SELECT source_id FROM news_bulletins WHERE "
                          "token = ?", (token,)).fetchone()
        source_id = owner[0] if owner else None
        superseded = c.execute(
            "SELECT token, audio_path FROM news_bulletins WHERE "
            "state = 'cached' AND token != ? AND source_id IS ?",
            (token, source_id)).fetchall()
        c.execute(
            "UPDATE news_bulletins SET state = 'expired', updated_at = ? "
            "WHERE state = 'cached' AND token != ? AND source_id IS ?",
            (_iso(now_dt), token, source_id))
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
    removed = 0
    for old_token, old_path in superseded:       # step 3: reclaim the slot
        if old_path and Path(old_path) != final:
            try:
                Path(old_path).unlink(missing_ok=True)
                removed += 1
            except OSError:
                logger.warning("news: could not remove superseded audio for "
                               "%s (cleanup will retry)", old_token[:8])
    return {"ok": True, "token": token, "expired": [t for t, _ in superseded],
            "removed_files": removed}


# --- the single-flight download pipeline ---------------------------------------
#
# ONE news download may run at a time, process-wide. The service is a single
# uvicorn worker (deploy/README, --workers 1), so a module lock is the whole
# coordination story; a second attempt fails fast rather than queueing, and
# the lock is released on every path including exceptions. A crashed process
# cannot wedge downloads: the lock dies with it.

_download_lock = threading.Lock()


def _transition(token: str, expected, new_state: str,
                now: datetime | None = None, db_path=None,
                **fields: Any) -> bool:
    """Conditional state move: exactly one caller can win."""
    sets = ", ".join(k + " = ?" for k in fields)
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE news_bulletins SET state = ?, updated_at = ?"
            + (", " + sets if sets else "")
            + " WHERE token = ? AND state IN ("
            + ",".join("?" for _ in expected) + ")",
            (new_state, _iso(now), *fields.values(), token, *expected))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def fetch_latest_bulletin(bulletins: dict[str, Any] | None = None,
                          now: datetime | None = None, db_path=None,
                          open_fn=None) -> dict[str, Any]:
    """Fetch, validate and activate the newest publisher bulletin.

    Returns {"ok": bool, "result"/"reason": str, ...}. Explicitly manual:
    nothing in Music Brain calls this on a timer, at startup, or from the DJ
    path. A failure of any kind leaves the previously cached bulletin, its
    file and its token exactly as they were."""
    now_dt = now or service_clock.legacy_now()
    # The one-slot cache deletes the previous MP3 once a replacement
    # activates. That is safe until the current token is the one Music
    # Assistant is about to play or is playing: replacing it would turn a
    # queued URL into a 410 mid-event. While a news event holds the
    # artifact, a refresh -- automatic or manual -- is refused rather than
    # allowed to pull the file out from under MA.
    try:
        import news_schedule
        locked, token = news_schedule.bulletin_locked()
    except Exception:
        locked, token = False, ""
    if locked:
        return {"ok": False, "reason": "bulletin_on_air", "token": token}
    if not _download_lock.acquire(blocking=False):
        return {"ok": False, "reason": "download_in_progress"}
    try:
        if bulletins is None:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        if (bulletins.get("mode") or ORIGIN_PUBLISHER) != ORIGIN_PUBLISHER:
            return {"ok": False, "reason": "mode_not_implemented"}
        try:
            source_id, feed_url, _label = resolve_publisher_source(bulletins)
        except FeedError as exc:
            return {"ok": False, "reason": "source_unconfigured",
                    "error": str(exc)}

        try:
            xml_bytes = fetch_feed_bytes(feed_url, open_fn)
            entries = parse_feed(xml_bytes, feed_url)
        except FeedError as exc:
            return {"ok": False, "reason": "feed_failed", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - network of any shape
            return {"ok": False, "reason": "feed_failed",
                    "error": str(exc)[:200]}

        entry, why = select_newest_viable(entries)
        if entry is None:
            return {"ok": False, "reason": why or "no_items"}

        row = upsert_publisher_entry(source_id, feed_url, entry, now_dt,
                                     db_path)
        token = row["token"]
        if (row["state"] == "cached" and row["audio_path"]
                and Path(row["audio_path"]).exists()):
            return {"ok": True, "result": "already_current", "token": token,
                    "title": row["title"], "duration_s": row["duration_s"]}

        if not _transition(token, ("discovered", "failed", "expired"),
                           "downloading", now_dt, db_path):
            return {"ok": False, "reason": "not_claimable", "token": token}

        tmp = _tmp_path(token)
        try:
            # Human-facing metadata goes in with the download itself, so no
            # consumer can ever see the token as a name and the file is
            # written once. Everything the tag says is known before the
            # first byte arrives; a download that fails validation is
            # removed, tag and all, exactly as before.
            pres = presentation_for(
                source_id, feed_url, entry.get("title") or "",
                ((bulletins or {}).get("source_name") or ""))
            size = _download_audio(entry["enclosure_url"], tmp, open_fn,
                                   tag=_presentation_tag(pres))
            duration = _validate_audio(tmp)
            # Trailing breath LAST, so the artifact is completely final
            # before anything can observe it: appending frames leaves the
            # front-mounted ID3 untouched. Both the duration and the size
            # are then re-read from the FINAL bytes rather than carried
            # over from the download.
            duration = append_tail_silence(tmp)
            if _program_audio_as_sound_effect():
                duration = level_for_sound_effect(tmp, token, pres, duration)
            size = tmp.stat().st_size
        except Exception as exc:  # noqa: BLE001 - the failure boundary
            tmp.unlink(missing_ok=True)
            _transition(token, ("downloading",), "failed", service_clock.legacy_now(),
                        db_path, error=str(exc)[:300])
            logger.warning("news bulletin download failed: %s", exc)
            return {"ok": False, "reason": "download_failed",
                    "error": str(exc)[:200], "token": token}

        try:
            activated = activate_artifact(token, tmp, size, duration, now_dt,
                                          db_path)
        except Exception as exc:  # noqa: BLE001 - e.g. a full disk at rename
            tmp.unlink(missing_ok=True)
            _transition(token, ("downloading",), "failed", service_clock.legacy_now(),
                        db_path, error=str(exc)[:300])
            logger.warning("news bulletin activation failed: %s", exc)
            return {"ok": False, "reason": "activation_failed",
                    "error": str(exc)[:200], "token": token}
        if not activated.get("ok"):
            tmp.unlink(missing_ok=True)
            return {"ok": False, "reason": activated.get("reason", "failed"),
                    "token": token}
        cleanup(now_dt, db_path)
        return {"ok": True, "result": "downloaded", "token": token,
                "title": row["title"], "guid": row["source_guid"],
                "published_at": row["published_at"], "size_bytes": size,
                "duration_s": duration, "expired": activated["expired"]}
    finally:
        _download_lock.release()


# --- serving gate + bounded cleanup --------------------------------------------

def audio_available(token: str, db_path=None) -> tuple[int, Path | None]:
    """(status, path) for the finite bulletin endpoint. A token resolves
    ONLY to its own file, never to 'whatever is current'. Unknown -> 404.

    Transport safety (program_audio): a KNOWN token that is no longer the
    current cached artifact (superseded, or its file already cleaned up) answers
    200 with the retired-silence tombstone instead of 410 -- an aired
    bulletin's queue item stays addressable in MA history forever, and a
    history replay must meet an ordinary short track, never a 410 storm
    (see program_audio). Stale NEWS is the clearest case of speech that
    must not replay: yesterday's headlines presented as current would be
    worse than the silence."""
    if not _TOKEN_RE.match(token or ""):
        return 404, None
    import program_audio
    import program_tokens
    stone = program_audio.tombstone()
    retired = (200, stone) if stone else (410, None)

    row = get_bulletin(token, db_path)
    if row is None:
        # A pruned row is not a token that never existed. Same reasoning as
        # the DJ gate: Music Assistant addresses a queue item forever, and
        # `cleanup` bounds this table.
        if program_tokens.is_retired(token, db_path):
            return retired
        return 404, None
    if row["state"] != "cached":
        return retired

    # SPENT BEATS CACHED. Retention keeps one bulletin per SOURCE, and it
    # keeps it for as long as that source has not fetched a newer one --
    # which is a statement about history, not about what may go on the air.
    # A bulletin that has finished airing is spent whatever its row still
    # says, and the gap is a full-length answer: an old
    # queue item for yesterday's afternoon bulletin, still `cached` because the
    # station had been idle overnight, would play yesterday's news
    # at full length as today's.
    if program_tokens.is_retired(token, db_path):
        return retired

    path = Path(row["audio_path"] or "")
    if not row["audio_path"] or path != _audio_path(token) or not path.exists():
        return retired

    # THE SECOND GATE, and it is deliberately not a duplicate of the first.
    # The spent ledger knows what THIS build completed; the clock knows what
    # is news. An installation upgrading into this code has cached rows that
    # predate the ledger entirely, and a bulletin old enough to be stale must
    # not become current again merely because nothing recorded its ending.
    # Same number the planner refuses to AIR a stale bulletin on, read from
    # it rather than restated so the two cannot drift.
    if _beyond_bulletin_age(row["published_at"]):
        return retired
    return 200, path


def _beyond_bulletin_age(published_at: str | None) -> bool:
    """Is this item too old to be presented as news at all?

    Fails OPEN on an unreadable date: a feed that gave no usable timestamp
    has always been allowed to air, and this gate exists to stop stale
    speech, not to invent a new refusal for an old shape of feed."""
    try:
        import news_schedule
        limit = float(news_schedule.MAX_BULLETIN_AGE_MIN)
    except Exception:  # noqa: BLE001 - the planner's number or nothing
        return False
    age = _age_minutes(published_at)
    return age is not None and age > limit


def own_audio_playable(token: str, db_path=None) -> bool:
    """Is THIS token's own recording still on disk and still the one its
    source retains?

    audio_available() deliberately answers 200 for a retired token, with a
    tombstone: an aired bulletin's queue row stays addressable in Music
    Assistant's history forever and must meet a short silence there, never a
    410 storm. That contract is right for the transport and wrong for a
    button: "Listen again" offered on a tombstone plays a few seconds of
    nothing and looks like a broken product.

    So the screen asks this question instead -- the row is cached AND the
    file it names is the file that belongs to it AND it is there -- and the
    transport keeps asking the other one. Neither answer is derived from the
    other, because they are not the same question."""
    if not _TOKEN_RE.match(token or ""):
        return False
    row = get_bulletin(token, db_path)
    if row is None or row["state"] != "cached":
        return False
    path = Path(row["audio_path"] or "")
    return bool(row["audio_path"]) and path == _audio_path(token) and path.exists()


def cleanup(now: datetime | None = None, db_path=None) -> dict[str, int]:
    """One-slot-per-source hygiene: the CACHED bulletin of each source keeps
    its MP3, every other file under news_audio/ is reclaimed, temp files do
    not accumulate, and metadata stays bounded. Never touches a cached row
    or anything outside news_audio/.

    The set of files to keep is simply the set of cached tokens: the switch
    in activate_artifact already guarantees at most one cached row per
    source, so this does not have to re-derive "latest" -- it only has to
    trust the state machine and delete everything the state machine does not
    name. That is also what makes it correct after a crash between the
    rename and the switch: a file whose row never became cached is an
    orphan no token can reach, and it goes.

    A PRUNED ROW LEAVES ITS RETIREMENT FACT BEHIND, in the same transaction
    that removes it -- the DJ reaper's discipline, applied here. A row is
    what makes a token KNOWN, and a known token must never become
    transport-fatal (program_audio). Every non-cached row this prunes is one
    whose token may already be sitting in a Music Assistant queue: a
    bulletin handed to MA that never aired is released by the planner, is
    superseded to 'expired' by the next episode, and reaches this cap in
    about two days -- at which point, without the fact, its queue item
    would start answering 404. So the fact is written first, on this
    connection, and if it cannot be written the row is not deleted either:
    the prune rolls back and is retried on the next call. Cached rows are
    never touched and never recorded -- recording one would mark a fresh
    bulletin spent, and the serving gate treats spent as final."""
    now_dt = now or service_clock.legacy_now()
    # File ages are measured on the filesystem's own clock -- real host
    # time -- never on a simulated service instant; a caller's explicit
    # `now` is still honoured.
    file_now = now.timestamp() if now is not None else time.time()
    stats = {"files": 0, "temps": 0, "rows": 0, "retired_facts": 0}
    c = _connect(db_path)
    try:
        keep_tokens = {row[0] for row in c.execute(
            "SELECT token FROM news_bulletins WHERE state = 'cached'"
        ).fetchall() if row[0]}
        try:
            import program_tokens
            victims = c.execute(
                "SELECT token FROM news_bulletins WHERE state != 'cached' "
                "ORDER BY updated_at DESC LIMIT -1 OFFSET ?",
                (ROW_CAP,)).fetchall()
            for (tok,) in victims:
                # THE FACT BEFORE THE FORGETTING. Same connection, same
                # transaction as the delete below: no crash can leave a
                # token whose row is gone and whose retirement was never
                # recorded.
                stats["retired_facts"] += program_tokens.record(
                    c, [tok], program_tokens.KIND_NEWS, now_dt)
                cur = c.execute("DELETE FROM news_bulletins WHERE token = ? "
                                "AND state != 'cached'", (tok,))
                stats["rows"] += cur.rowcount
            c.commit()
        except Exception:  # noqa: BLE001 - a row is never forgotten alone
            c.rollback()
            stats["rows"] = 0
            stats["retired_facts"] = 0
            logger.warning("news: row prune rolled back; no row is deleted "
                           "without its retirement fact (retried next call)",
                           exc_info=True)
    finally:
        c.close()

    adir = news_audio_dir()
    if not adir.is_dir():
        return stats
    for path in adir.iterdir():
        name = path.name
        try:
            if name.startswith(".news-") and name.endswith(".tmp"):
                age = file_now - path.stat().st_mtime
                if age > TMP_GRACE_S:   # never reap a live download's temp
                    path.unlink(missing_ok=True)
                    stats["temps"] += 1
            elif name.startswith("news-") and name.endswith(".mp3"):
                token = name[len("news-"):-len(".mp3")]
                if not _TOKEN_RE.match(token):
                    logger.warning("news: unrecognized file left in place: %s",
                                   name)
                    continue
                if token not in keep_tokens:
                    path.unlink(missing_ok=True)
                    stats["files"] += 1
        except OSError:
            continue
    return stats


# --- presentation (what a human and the MA queue see) --------------------------

def presentation_for(source_id: str, feed_url: str = "",
                     entry_title: str = "",
                     source_name: str = "") -> dict[str, str]:
    """Human-facing title/artist/artwork for a bulletin.

    source_name, when the station owner supplied one, outranks everything:
    it is their word for their newsroom. Otherwise a built-in carries its
    curated presentation and any other feed falls back to the item's own
    title and its host -- never 'custom:<hash>', which is internal
    identity, and never the token, which is a credential.

    The entry title is BULLETIN METADATA and nothing more. It names this
    episode in a queue and in an ID3 tag; it is never handed to the DJ as
    a fact about the world, because a podcast episode title is not a
    headline somebody reported."""
    name = (source_name or "").strip()
    entry = BUILTIN_SOURCES.get(source_id)
    if entry:
        return {"title": entry.get("title") or entry["label"],
                "artist": name or entry.get("artist") or entry["label"],
                "label": name or entry["label"],
                "artwork": entry.get("artwork") or ""}
    shown = name or source_name_from_feed(feed_url) or "Newsroom"
    return {"title": (entry_title or "News Bulletin").strip(),
            "artist": shown,
            "label": shown,
            "artwork": ""}


def _program_audio_as_sound_effect() -> bool:
    try:
        import program_loudness
        return program_loudness.enabled()
    except Exception:  # noqa: BLE001 - the track path is the safe answer
        return False


def level_for_sound_effect(tmp: Path, token: str, pres: dict[str, str],
                           duration: float) -> float:
    """Level a downloaded bulletin, in its temp slot, for the sound-effect
    path, and return its final duration.

    Music Assistant does not normalise a sound effect, and publisher
    bulletins arrive anywhere from about -25 to -17 LUFS. So with the
    prototype on, this one step re-encodes the publisher's audio: one
    measured gain, plus a true-peak limiter (program_loudness). The content
    is unchanged, sponsorship included. This is the processing MA's own
    normaliser applies to a bulletin that airs as a track, done before the
    file is served instead of while it plays. With the prototype off,
    nothing here runs and the publisher's frames stay byte for byte.

    It runs BEFORE activation, so no URL serves these bytes yet. The
    levelled result is staged beside the temp file and swapped in only once
    it probes. On any failure the temp file is exactly as it was, and the
    bulletin is inserted as a track, which MA normalises."""
    import program_loudness as PL
    staged = tmp.with_name(tmp.name[:-len(".tmp")] + ".lvl.tmp")
    try:
        frames, report = PL.level(tmp.read_bytes())
        tag = _presentation_tag(pres, {PL.MARKER_DESC: PL.marker(report)})
        with staged.open("wb") as f:
            f.write(tag + frames)
            f.flush()
            os.fsync(f.fileno())
        final = _probe_duration(staged)
        if final is None:
            raise PL.LevelError(PL.R_TOOL_FAILED,
                                "levelled bulletin does not probe", report)
        os.replace(staged, tmp)
    except Exception as exc:  # noqa: BLE001 - the unlevelled bulletin airs
        staged.unlink(missing_ok=True)
        PL.log_outcome("news", token, None, exc)
        return duration
    PL.log_outcome("news", token, report)
    return final


def _presentation_tag(pres: dict[str, str],
                      user_text: dict[str, str] | None = None) -> bytes:
    """The human-facing ID3v2 block a bulletin is written with, so the
    artifact is never served or queued as '[unknown]'. Best effort, exactly
    like the break tagger: a tag that cannot be built must not cost a
    perfectly good bulletin, so the download proceeds untagged. Only
    metadata -- the publisher's audio, sponsorship included, is never
    altered, which is why the tag is built by hand rather than by remuxing
    through ffmpeg. (The one exception is the sound-effect prototype's
    loudness step, level_for_sound_effect, which is off by default.)
    `user_text` carries that step's marker."""
    try:
        import audio_probe
        return audio_probe.build_tag(user_text, title=pres["title"],
                                     artist=pres["artist"],
                                     album=pres["label"])
    except Exception:                   # noqa: BLE001 - best effort by design
        logger.debug("bulletin ID3 tag could not be built", exc_info=True)
        return b""


# Breathing room at the END of a publisher bulletin. Without it a bulletin
# that ends on its last syllable cuts straight into music; this is the
# pause a human board operator would leave.
NEWS_BULLETIN_TAIL_S = float(os.environ.get("NEWS_BULLETIN_TAIL_S", "0.9"))


def _mp3_shape(path: Path) -> dict[str, Any] | None:
    """(sample_rate, channels, bitrate, xing) of an MP3, or None."""
    import audio_probe
    info = audio_probe.stream_info(path)
    if info is None:
        return None
    with path.open("rb") as f:
        head = f.read(8192)
    # A Xing/Info header states the frame count and duration up front.
    # Appending frames would NOT update it, so the file would then lie
    # about its own length. Read straight off the bytes, as it always was:
    # this is a question about what is in the file, not about what a
    # decoder makes of it.
    return {**info, "xing": (b"Xing" in head or b"Info" in head)}


def _silence_frames(shape: dict[str, Any], seconds: float) -> bytes:
    """`seconds` of digital silence encoded to match this bulletin.

    Deliberately NOT the DJ's MP3_ARGS: that contract is mono 24kHz 64k for
    the DJ's voice, while a publisher bulletin is whatever the publisher
    chose (often stereo 44.1kHz 128k CBR). The parameters are read off the
    downloaded file so a different feed still gets matching frames.

    The output carries no Xing header, no ID3 and no metadata -- pure MPEG
    frames, which is what makes them safe to concatenate."""
    import subprocess
    rate = int(shape.get("sample_rate") or 44100)
    channels = int(shape.get("channels") or 2)
    kbps = max(32, int((shape.get("bitrate") or 128000) // 1000))
    layout = "stereo" if channels >= 2 else "mono"
    try:
        proc = _run_silence(subprocess, rate, layout, kbps, seconds)
    except FileNotFoundError as exc:
        # No ffmpeg on this host at all. That is an environment capability,
        # not a damaged artifact, and it must not cost the listener the
        # bulletin itself -- see append_tail_silence.
        raise _NoSilenceTool(str(exc)) from exc
    if proc.returncode != 0 or not proc.stdout:
        raise FeedError("silence generation failed: %r" % (proc.stderr[:200],))
    out = proc.stdout
    if out[:3] == b"ID3" or not (out[0] == 0xFF and (out[1] & 0xE0) == 0xE0):
        raise FeedError("generated silence is not raw MPEG frames")
    return out


class _NoSilenceTool(RuntimeError):
    """ffmpeg is not installed here. Distinct from a padding FAILURE."""


def _run_silence(subprocess, rate, layout, kbps, seconds):
    """One bounded ffmpeg call. argv form only, never shell=True."""
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl={layout}",
         "-t", f"{seconds:g}", "-codec:a", "libmp3lame", "-b:a", f"{kbps}k",
         "-write_xing", "0", "-id3v2_version", "0", "-map_metadata", "-1",
         "-f", "mp3", "pipe:1"],
        capture_output=True, timeout=60)


def append_tail_silence(path: Path, seconds: float | None = None) -> float:
    """Give the bulletin its trailing breath and return the FINAL duration.

    THE PUBLISHER'S AUDIO IS NEVER RE-ENCODED. Matching silent MPEG frames
    are appended to the existing bytes, so every original frame -- content,
    sponsorship, the lot -- survives byte-for-byte, and the ID3v2 tag at
    the front is equally untouched. This is possible because the artifact
    is CBR with no Xing header, so its length is derived from byte count
    and grows correctly when frames are added.

    A source that DOES carry a Xing/Info header is left unpadded on
    purpose: appending would produce a file whose own header understates
    its length. That is a deliberate, logged degradation -- the bulletin
    still airs, just without the breath -- and never a corrupt artifact.
    A real failure (ffmpeg, or a final artifact that will not probe) raises,
    and the caller abandons the fetch with the previous bulletin intact."""
    seconds = NEWS_BULLETIN_TAIL_S if seconds is None else seconds
    shape = _mp3_shape(path)
    if shape is None:
        raise FeedError("bulletin is not a probeable MP3 before padding")
    if seconds <= 0:
        return shape["length"]
    if shape["xing"]:
        logger.info("bulletin carries a Xing header; leaving it unpadded "
                    "rather than making it misreport its length")
        return shape["length"]
    try:
        silence = _silence_frames(shape, seconds)
    except _NoSilenceTool as exc:
        # ffmpeg is not installed on this host. The bulletin is perfectly
        # good, just unpadded; refusing it would cost the listener the news
        # over a missing dev-machine binary. A real padding FAILURE still
        # raises below and abandons the fetch.
        logger.warning("no ffmpeg available; bulletin left unpadded: %s", exc)
        return shape["length"]
    with path.open("ab") as f:          # append: nothing is rewritten
        f.write(silence)
    final = _probe_duration(path)
    if final is None:
        raise FeedError("padded bulletin no longer probes as MP3")
    if final <= shape["length"]:
        raise FeedError("padding did not extend the bulletin")
    return final


def audio_url(token: str) -> str:
    """The externally fetchable URL of a bulletin artifact, built from the
    same base the DJ audio path uses so one deployment setting governs
    both."""
    import config as _config
    return f"{_config.current().news_audio_url_base}/{token}.mp3"


def _presented(b: dict[str, Any],
               bulletins: dict[str, Any] | None) -> dict[str, Any]:
    """One cached row as the rest of the system sees it. Never exposes
    audio_path or the enclosure URL."""
    # The source's OWN configured name -- but only from a document that
    # really is multi-source. Reading the legacy `source_name` for every row
    # would label a second newsroom's bulletin with the first newsroom's
    # name; going the other way and reading a MIGRATED source's name on a
    # single-feed document would be worse, because the migration fills that
    # name in from the registry label and would quietly outrank a built-in
    # entry's own presentation. So: a stored `sources` list is the only
    # thing that switches this on, and every legacy document keeps exactly
    # the presentation it had.
    name = ""
    listed = (bulletins or {}).get("sources")
    if isinstance(listed, list) and listed:
        try:
            import news_sources
            src = news_sources.by_id(bulletins, b["source_id"])
            name = (src or {}).get("name") or ""
        except Exception:  # noqa: BLE001 - presentation never breaks reporting
            name = ""
    if not name:
        name = (bulletins or {}).get("source_name") or ""
    pres = presentation_for(b["source_id"], b["feed_url"], b["title"], name)
    return {"token": b["token"], "origin_type": b["origin_type"],
            "source_id": b["source_id"], "source_label": pres["label"],
            "display_title": pres["title"], "display_artist": pres["artist"],
            "artwork": pres["artwork"], "item_title": b["title"],
            "published_at": b["published_at"], "produced_at": b["produced_at"],
            "size_bytes": b["size_bytes"], "duration_s": b["duration_s"],
            "state": b["state"], "audio_url": audio_url(b["token"])}


def _settings_bulletins(bulletins: dict[str, Any] | None) -> dict[str, Any]:
    if bulletins is not None:
        return bulletins
    try:
        import settings
        return settings.load().get("bulletins") or {}
    except Exception:  # noqa: BLE001 - a label is never worth a failure
        return {}


def current_bulletin(db_path=None,
                     bulletins: dict[str, Any] | None = None,
                     source_id: str | None = None) -> dict[str, Any] | None:
    """The cached bulletin, presentation included. None when there is none.

    `source_id` asks for ONE SOURCE's slot and is what every caller that
    cares which newsroom it is talking about must pass: since the retention
    rule became per-source there can be several cached rows at once, and
    "the newest cached row of any source" is only the right answer to a
    question about the station as a whole. Without it the newest cached row
    overall is returned, which is what the existing single-feed callers
    (the developer bench, the legacy status block) mean.

    The presentation is recomputed from CURRENT settings rather than read
    back from the row: renaming the source in the Newsroom should rename
    what the Newsroom shows, without re-downloading anything. The ID3 the
    artifact already carries is the one it was stamped with, and is left
    alone -- rewriting a file Music Assistant may be about to read is not
    worth a label."""
    where = "state = 'cached'"
    args: tuple[Any, ...] = ()
    if source_id:
        where += " AND source_id = ?"
        args = (source_id,)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_COLS)} FROM news_bulletins WHERE {where} "
            "ORDER BY updated_at DESC LIMIT 1", args).fetchone()
    finally:
        c.close()
    if row is None:
        return None
    return _presented(_row_dict(row), _settings_bulletins(bulletins))


def current_bulletins(db_path=None,
                      bulletins: dict[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
    """EVERY source's retained bulletin, newest first -- the list the
    Newsroom's wire is built from.

    One row per source by construction, because activate_artifact leaves at
    most one cached row per source. A source that has never fetched is
    simply absent here; the Newsroom still shows it, from its configuration,
    which is the whole point of listing sources rather than bulletins."""
    c = _connect(db_path)
    try:
        rows = c.execute(
            f"SELECT {', '.join(_COLS)} FROM news_bulletins WHERE "
            "state = 'cached' ORDER BY updated_at DESC").fetchall()
    finally:
        c.close()
    cfg = _settings_bulletins(bulletins)
    return [_presented(_row_dict(r), cfg) for r in rows]


def sources_status(bulletins: dict[str, Any] | None,
                   errors: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Each configured bulletin source, with its own last outcome.

    Imported late and guarded: this is a reporting convenience, and a
    Newsroom that cannot describe itself must still be a Newsroom that
    airs. Never raises.
    """
    try:
        import news_schedule
        import news_sources
        state = news_schedule.load_state()
        out = []
        for src in news_sources.sources_of(bulletins):
            rec = news_schedule.source_record(state, src["id"])
            out.append({
                **src,
                "display_name": news_sources.display_name(src),
                "last_aired_token": rec.get("last_aired_token"),
                "last_aired_at": rec.get("last_aired_at"),
                "last_result": rec.get("last_result"),
                "last_result_at": rec.get("last_result_at"),
                "error": (errors or {}).get(src["id"]),
            })
        return out
    except Exception:  # noqa: BLE001 - reporting must not break airing
        logger.warning("bulletin source status unavailable", exc_info=True)
        return []


def status(bulletins: dict[str, Any] | None = None,
           db_path=None) -> dict[str, Any]:
    """Compact state for the developer UI: effective configuration, the one
    cached bulletin, and whether a download is running right now."""
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            bulletins = {}
    pub = (bulletins or {}).get("publisher") or {}
    source_id = label = feed_url = ""
    source_error = None
    try:
        source_id, feed_url, label = resolve_publisher_source(bulletins)
    except FeedError as exc:
        source_error = str(exc)
    sched = _schedule_status(bulletins)
    last_error = None
    source_errors: dict[str, str] = {}
    c = _connect(db_path)
    try:
        row = c.execute(
            "SELECT error FROM news_bulletins WHERE state = 'failed' "
            "ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row and row[0]:
            last_error = row[0][:200]
        # EACH source's own newest failure. The single `last_error` above is
        # the station's most recent trouble whoever caused it, which reads
        # on a two-newsroom screen as though both are broken; this is the
        # one that belongs beside a particular source's name.
        for sid, err in c.execute(
                "SELECT source_id, error FROM news_bulletins WHERE "
                "state = 'failed' AND error IS NOT NULL "
                "ORDER BY updated_at ASC").fetchall():
            if sid:
                source_errors[str(sid)] = str(err)[:200]
        rows = c.execute("SELECT COUNT(*) FROM news_bulletins").fetchone()[0]
    finally:
        c.close()
    return {
        "enabled": bool((bulletins or {}).get("enabled")),
        "mode": (bulletins or {}).get("mode") or ORIGIN_PUBLISHER,
        # The STORED name, beside the resolved label below. The settings
        # form needs to know the difference between "the owner typed
        # nothing" and "the owner typed the host".
        "source_name": ((bulletins or {}).get("source_name") or ""),
        "publisher": {"source": pub.get("source") or "",
                      "custom_feed_url": pub.get("custom_feed_url") or "",
                      "source_id": source_id, "label": label,
                      "feed_url": feed_url, "error": source_error},
        "builtin_sources": [{"id": k, "label": v["label"]}
                            for k, v in sorted(BUILTIN_SOURCES.items())],
        # EVERY configured source, with what each last managed. The single
        # `publisher` block above is the one whose turn is next and stays
        # for the existing consumers; this is the list the Newsroom shows.
        "sources": sources_status(bulletins, source_errors),
        # No bulletins argument: current_bulletin loads settings itself when
        # it needs them, and every existing caller (and every test stub)
        # passes db_path alone. Widening this call shape for a label would
        # be a poor trade.
        "current": current_bulletin(db_path),
        # EVERY source's retained bulletin. `current` above is the single
        # newest one and stays for the developer bench and the legacy
        # consumers; this is the list the Newsroom's wire is built from, and
        # it is the reason a second newsroom no longer vanishes from the
        # screen the moment the first one refreshes.
        "currents": current_bulletins(db_path, bulletins),
        "download_in_progress": _download_lock.locked(),
        "last_error": last_error,
        "rows": rows,
        # ONE authoritative automation value. The key survives for
        # backward compatibility, but the old literal "not_implemented" has
        # been retired: on-air automation now exists, and reporting
        # otherwise while a bulletin could be queued would simply be false.
        # It mirrors schedule.automation, which is the real state.
        "automation": sched.get("automation", "unavailable"),
        "schedule": sched,
    }


def _schedule_status(bulletins: dict[str, Any] | None) -> dict[str, Any]:
    """Planning view, additively. Read-only and fenced: the status page must
    never be the thing that breaks, and it must never trigger planning."""
    try:
        import news_schedule
        return news_schedule.schedule_status(bulletins)
    except Exception:
        logger.warning("news schedule status unavailable", exc_info=True)
        return {"automation": "unavailable"}


def test_feed(bulletins: dict[str, Any] | None = None,
              open_fn=None, now: datetime | None = None) -> dict[str, Any]:
    """Network + parse only: prove a feed is reachable and usable. Creates
    no row, downloads no audio, writes no file, changes no settings.

    Every outcome is reported as facts a person can act on -- reachable,
    parsed, how many items, what the newest one is, how old it is, whether
    it carries playable audio -- and a failure is a refusal to report, not
    an exception. Nothing here can reach playback: this call has no Music
    Assistant client, writes no artifact and touches no queue."""
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            bulletins = {}
    if (bulletins or {}).get("mode", ORIGIN_PUBLISHER) != ORIGIN_PUBLISHER:
        return {"ok": False, "reason": "mode_not_implemented"}
    try:
        source_id, feed_url, label = resolve_publisher_source(bulletins)
    except FeedError as exc:
        return {"ok": False, "reason": "source_unconfigured",
                "error": str(exc)}
    where = {"source_id": source_id, "label": label, "feed_url": feed_url}
    try:
        xml_bytes = fetch_feed_bytes(feed_url, open_fn)
    except PayloadTooLarge as exc:
        # The server answered; the body was simply bigger than the cap.
        # Reporting that as "unreachable" is a false statement about a
        # working address, and it is the one outcome a person cannot act
        # on -- so it gets its own reason and keeps reachable=True.
        return {"ok": False, "reason": "feed_too_large", "reachable": True,
                "parsed": False, "max_bytes": FEED_MAX_BYTES,
                "error": str(exc)[:200], **where}
    except Exception as exc:  # noqa: BLE001 - network of any shape
        # Nothing came back at all: this is reachability, and saying so is
        # the difference between "check the address" and "check the feed".
        return {"ok": False, "reason": "unreachable", "reachable": False,
                "parsed": False, "error": str(exc)[:200], **where}
    try:
        entries = parse_feed(xml_bytes, feed_url)
    except FeedError as exc:
        return {"ok": False, "reason": "unparseable", "reachable": True,
                "parsed": False, "error": str(exc)[:200], **where}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "unparseable", "reachable": True,
                "parsed": False, "error": str(exc)[:200], **where}
    entry, why = select_newest_viable(entries)
    base = {"source_id": source_id, "label": label, "feed_url": feed_url,
            "items": len(entries), "reachable": True, "parsed": True}
    if entry is None:
        return {"ok": False, "reason": why or "no_items", **base}
    pres = presentation_for(source_id, feed_url, entry.get("title") or "",
                            ((bulletins or {}).get("source_name") or ""))
    return {"ok": True, **base,
            # item_title is the EPISODE's own name -- bulletin metadata for
            # a queue row and an ID3 tag. It is reported here so a person
            # can see the test found the right feed, and it is deliberately
            # not called a headline anywhere, because it is not one.
            "item_title": entry.get("title"),
            "display_title": pres["title"], "display_artist": pres["artist"],
            "published_at": entry.get("published_at"),
            "published_age_minutes": _age_minutes(entry.get("published_at"), now),
            "content_type": entry.get("content_type"),
            "expected_length": entry.get("expected_length"),
            "duration_s": entry.get("duration_s"),
            "has_enclosure": bool(entry.get("enclosure_url")),
            "guid": (entry.get("source_guid") or "")[:40]}


def _age_minutes(published_at: str | None,
                 now: datetime | None = None) -> float | None:
    """How old this item is, in minutes, or None when the feed gave no
    usable date. Never negative-clamped: a feed dated in the future is a
    fact about that feed and the Newsroom should be able to say so."""
    if not published_at:
        return None
    try:
        when = datetime.fromisoformat(published_at)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    ref = now or service_clock.now()
    if ref.tzinfo is None:
        # A naive `now` is a SERVICE-CLOCK stamp, and under Supervisor the
        # add-on inherits the home's timezone -- so relabelling it as UTC
        # shifts every age by the UTC offset. Interpret it in the system
        # zone instead. (news_schedule's two-clocks note: the same mistake.)
        ref = ref.astimezone(timezone.utc)
    return (ref - when).total_seconds() / 60.0
