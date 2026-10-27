#!/usr/bin/env python3
"""The service clock: the ONE answer to "what instant is it?" for every
domain decision -- due, expired, aged, cut off, stamped for later aging.

Four kinds of time live in this runtime, and only one of them is here:

  service instant   now(): timezone-aware UTC. Host time in production;
                    host time plus a forward-only offset in the Accelerated
                    Lab; an installed source in tests.
  station civil     dj_context.station_now()/station_at(): the instant seen
                    in the station's IANA zone. Daypart, shifts, the spoken
                    clock, dates, minute-of-hour. Never the container's zone.
  monotonic         time.monotonic(), called directly. Timeouts, latency,
                    throttles, locks. NEVER simulated, never routed here.
  real host time    time.time() / datetime.now(timezone.utc), called
                    directly: Music Assistant and Home Assistant epoch
                    comparisons, file mtimes, write-only audit records.
                    NEVER simulated, never routed here.

legacy_now() is not a fifth clock. The persisted stamp families (DJ breaks,
playback sessions, News, context state) store NAIVE ISO strings in the
process's own zone and compare them lexically; that encoding is kept as it
is, so legacy_now() is the service instant written in that encoding -- and
with no lab offset and no test source it is literally datetime.now(), fold
and all. It carries no civil meaning: civil questions go to dj_context.

The lab offset exists only when PARLORWAVE_LAB_CLOCK=1 is in the process
environment at start. Without it the offset is zero, the persisted lab file
is never read, and lab_advance() refuses. The offset only ever moves
forward: a stamp written on a later simulated instant must never end up in
the future of the clock that ages it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger("musicbrain")

LAB_ENV = "PARLORWAVE_LAB_CLOCK"
LAB_FILE = "lab_clock.json"

_lock = threading.Lock()
#: Test seam: a callable returning the service instant (aware). None in
#: production. Installed only through installed().
_source: Callable[[], datetime] | None = None
#: Lab offset. Zero unless the lab is enabled AND has advanced.
_offset = timedelta(0)
_lab_meta: dict[str, Any] = {}


def lab_enabled() -> bool:
    """Lab clock controls exist only for a process started with
    PARLORWAVE_LAB_CLOCK=1. Read live so a test can scope it; production
    never sets it."""
    return os.environ.get(LAB_ENV) == "1"


# --- the instant ---------------------------------------------------------------

def now() -> datetime:
    """The service instant, timezone-aware UTC."""
    src = _source
    if src is not None:
        return _utc(src())
    real = datetime.now(timezone.utc)
    return real + _offset if _offset else real


def legacy_now() -> datetime:
    """The service instant in the LEGACY persisted encoding: naive, in the
    process's own zone, fold set in a repeated hour. Exactly datetime.now()
    when nothing is installed and the offset is zero."""
    if _source is None and not _offset:
        return datetime.now()
    return to_legacy(now())


def offset() -> timedelta:
    return _offset


def to_legacy(instant: datetime) -> datetime:
    """An instant in the legacy encoding. fromtimestamp is the code path
    datetime.now() itself takes, so fold is set exactly as it would be."""
    if instant.tzinfo is None:
        return instant                       # already legacy
    return datetime.fromtimestamp(instant.timestamp())


def from_legacy(value: Any) -> datetime | None:
    """A stamp (datetime or ISO string) as an aware UTC instant, or None.
    Naive values are the legacy encoding: resolved in the process zone,
    honouring their fold (a parsed string has fold=0, the earlier reading).
    """
    if isinstance(value, str):
        if not value:
            return None
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    try:
        return _utc(value)
    except (OverflowError, OSError, ValueError):
        return None


def instant(value: datetime | None = None) -> datetime:
    """`value` as an aware UTC instant; the service instant when None."""
    if value is None:
        return now()
    return _utc(value)


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.astimezone()                 # legacy: the process zone, fold
    return dt.astimezone(timezone.utc)


# --- DST-safe aging of legacy stamps ----------------------------------------------
#
# A legacy stamp is a naive wall reading in the process zone. Outside a DST
# transition it names exactly one instant. A stamp written during the
# REPEATED fall-back hour names two (its first and second passes, an hour
# apart), and nothing on disk says which. These helpers never pretend to
# know: they work with both readings and choose per decision.
#
#   destructive (revoke, abandon, expire, delete, mark missed, take over a
#   claim): act only when the stamp is certainly old enough. Never early;
#   at most one ambiguous hour late.
#   release (gaps, cooldowns, freshness, repetition guards): hold only while
#   the stamp is certainly recent. Never negative; never held longer than
#   the truth; possibly released up to one ambiguous hour early.
#   spoken ages: only when the age is exact (age_exact), else nothing.
#
# Outside a transition every helper returns exactly what the old naive
# arithmetic did, byte for byte.

def readings(stamp: Any) -> tuple[datetime, datetime] | None:
    """(earliest, latest) aware instants a stamp can denote, or None. An
    aware stamp, or a legacy one outside a transition, gives one instant
    twice."""
    if isinstance(stamp, str):
        try:
            stamp = datetime.fromisoformat(stamp)
        except ValueError:
            return None
    if not isinstance(stamp, datetime):
        return None
    try:
        if stamp.tzinfo is not None:
            one = stamp.astimezone(timezone.utc)
            return one, one
        a = _utc(stamp.replace(fold=0))
        b = _utc(stamp.replace(fold=1))
    except (OverflowError, OSError, ValueError):
        return None
    return (a, b) if a <= b else (b, a)


def age_bounds(stamp: Any, now: datetime | None = None
               ) -> tuple[float, float] | None:
    """(lo, hi) seconds the stamp can be old at `now` (a legacy `now` is
    read honouring its fold). lo == hi except for a stamp written in a
    repeated hour. None for an unreadable stamp."""
    r = readings(stamp)
    if r is None:
        return None
    n = instant(now)
    return (n - r[1]).total_seconds(), (n - r[0]).total_seconds()


def age_exact(stamp: Any, now: datetime | None = None) -> float | None:
    """Seconds since the stamp when that is KNOWN, else None -- for words
    a listener hears, which must never be a guess."""
    b = age_bounds(stamp, now)
    return b[0] if b is not None and b[0] == b[1] else None


def legacy_shift(base: datetime | None, delta: timedelta) -> datetime:
    """The legacy reading of instant(base) + delta. When no transition lies
    in between it is exactly `base + delta` (the old arithmetic, so stamps
    and cutoffs stay byte-identical); across one it is the true wall
    reading, fold set."""
    base = legacy_now() if base is None else base
    target = instant(base) + delta
    if base.tzinfo is None:
        cand = base + delta
        try:
            if (_utc(cand.replace(fold=0)) == target
                    and _utc(cand.replace(fold=1)) == target):
                return cand
        except (OverflowError, OSError, ValueError):
            pass
    return to_legacy(target)


def legacy_iso_at(at: datetime) -> str:
    """An instant (aware, or a legacy reading) as a legacy stamp string."""
    return to_legacy(instant(at)).isoformat(timespec="seconds")


def _cutoff(now: datetime | None, delta: timedelta, destructive: bool) -> str:
    base = legacy_now() if now is None else now
    point = legacy_shift(base, -delta)
    r = readings(point)
    if r is not None and r[0] != r[1]:
        gap = r[1] - r[0]
        at = instant(point)
        if destructive and point.fold == 0:
            # the cutoff's wall reading also names a LATER instant: only
            # what precedes the whole repeated hour is certainly older
            point = to_legacy(at - gap)
        elif not destructive and point.fold == 1:
            # ...or an EARLIER one: only what follows the repeated hour is
            # certainly newer
            point = to_legacy(at + gap)
    return point.isoformat(timespec="seconds")


def cutoff_destructive(now: datetime | None, delta: timedelta) -> str:
    """A legacy-stamp string W for `stamp < W` destructive SQL: every stamp
    that compares below W is CERTAINLY older than `delta`. Exactly the old
    `_iso(now - delta)` outside a transition."""
    return _cutoff(now, delta, destructive=True)


def cutoff_release(now: datetime | None, delta: timedelta) -> str:
    """A legacy-stamp string W for `stamp > W` hold-off SQL: every stamp
    that compares above W is CERTAINLY younger than `delta`. Exactly the
    old `_iso(now - delta)` outside a transition."""
    return _cutoff(now, delta, destructive=False)


# --- tests ---------------------------------------------------------------------

@contextmanager
def installed(source: Callable[[], datetime]) -> Iterator[None]:
    """Install a service-instant source for the duration of a block. The
    source must return aware datetimes (naive ones are read as legacy)."""
    global _source
    with _lock:
        previous = _source
        _source = source
    try:
        yield
    finally:
        with _lock:
            _source = previous


class FakeClock:
    """A settable instant for tests: clock = FakeClock(start); with
    service_clock.installed(clock): ... clock.advance(seconds=60)."""

    def __init__(self, start: datetime) -> None:
        self.at = _utc(start)

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **delta: float) -> datetime:
        self.at = self.at + timedelta(**delta)
        return self.at


# --- the Accelerated Lab ---------------------------------------------------------

def _lab_path() -> Path:
    import paths
    return Path(paths.BASE_DIR) / LAB_FILE


def lab_status() -> dict[str, Any]:
    real = datetime.now(timezone.utc)
    service = now()
    return {"enabled": lab_enabled(),
            "offset_s": _offset.total_seconds(),
            "real_utc": real.isoformat(timespec="seconds"),
            "service_utc": service.isoformat(timespec="seconds"),
            "run_id": _lab_meta.get("run_id"),
            "advanced_at_real": _lab_meta.get("advanced_at_real")}


def lab_advance(seconds: float) -> dict[str, Any]:
    """Move the service instant FORWARD by `seconds` and persist the new
    offset, so a restart mid-run resumes the same timeline. Refused unless
    the lab is enabled; refused for anything but a positive finite step."""
    global _offset, _lab_meta
    if not lab_enabled():
        raise PermissionError("lab clock is not enabled in this process")
    try:
        step = float(seconds)
    except (TypeError, ValueError):
        raise ValueError("seconds must be a number") from None
    if not step > 0 or step == float("inf"):
        raise ValueError("the lab clock only moves forward")
    with _lock:
        new = _offset + timedelta(seconds=step)
        meta = {"offset_s": new.total_seconds(),
                "run_id": _lab_meta.get("run_id")
                or datetime.now(timezone.utc).strftime("lab-%Y%m%dT%H%M%SZ"),
                "advanced_at_real":
                    datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write_lab(meta)
        _offset, _lab_meta = new, meta
    logger.warning("LAB CLOCK advanced %+.0fs; service time is now %s "
                   "(offset %+.0fs)", step,
                   now().isoformat(timespec="seconds"), new.total_seconds())
    return lab_status()


def _write_lab(meta: dict[str, Any]) -> None:
    path = _lab_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".lab_clock-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_lab() -> None:
    """Resume a persisted lab offset. A no-op (the file is not even read)
    unless the lab is enabled. Called once at import, before any module
    that captures a start instant reads the clock."""
    global _offset, _lab_meta
    if not lab_enabled():
        return
    try:
        meta = json.loads(_lab_path().read_text(encoding="utf-8"))
        value = float(meta.get("offset_s") or 0.0)
    except FileNotFoundError:
        return
    except Exception:  # noqa: BLE001 - a corrupt lab file is a lab problem
        logger.warning("LAB CLOCK file unreadable; offset stays zero",
                       exc_info=True)
        return
    if value > 0:
        _offset, _lab_meta = timedelta(seconds=value), dict(meta)
        logger.warning("LAB CLOCK resumed with offset %+.0fs", value)


load_lab()
