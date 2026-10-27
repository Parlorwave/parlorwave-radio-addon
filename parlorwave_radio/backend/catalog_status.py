#!/usr/bin/env python3
"""Is there a local Parlorwave music catalog the autonomous station tools
can actually work from? READ ONLY.

THE PRODUCT QUESTION. Base Parlorwave Radio does not need a catalog at
all: a listener starts an album, a playlist or a station in Music
Assistant, Parlorwave adopts that session, Music Assistant stays the
queue and programming authority, and Parlorwave adds DJ breaks and news
between the tracks. Nothing in that path reads this module.

A SECOND, OPTIONAL set of features does need one, because each of them
picks the music itself:

    Start a station from this song      Endless Radio refill
    Request / Play it                   Station feel, artist rotation
    autonomous station building         thumbs taste-learning

Until now there was no public signal for whether those are worth
offering, so the surfaces guessed -- from a station being active, from a
file existing, from unrelated health state. This module is the one
authoritative answer, and every gate reads it.

`usable` is NOT "the catalog table is non-empty". It is the threshold the
station builder itself refuses below: station_radio.MIN_RESOLVED, the
number of tracks that must survive selection and resolution before a
station is written to a queue at all. Fewer than that and every
catalog-dependent control would present a button that cannot work. The
constant is IMPORTED, never restated, so the promise the capability makes
and the refusal the builder performs can never drift apart.

NO CATALOG IS A SUPPORTED STATE, NOT A PROBLEM. Nothing here produces a
health problem, a warning or a severity; `health.report` carries this as a
plain feature fact beside the others. An install with no catalog is a
correctly configured install.

Nothing here reveals where a catalog lives: no path, no filename, no
source. A count and a verdict is the whole public surface.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

# The single source of truth, imported rather than repeated. station_radio
# refuses to write a station when fewer than this many picks resolve; a
# catalog that cannot supply that many cannot produce one either.
from station_radio import MIN_RESOLVED as MINIMUM_TRACKS  # noqa: E402

# The stable machine-readable refusal every catalog-dependent surface uses.
REASON = "no_local_catalog"

# The one sentence a customer is shown when a catalog-dependent action is
# attempted anyway. It says what is true and what to do instead; it does
# not apologise, and it does not describe setup this release does not
# ship.
MESSAGE = ("This installation has no local ParlorWave music catalog, so "
           "ParlorWave Radio cannot pick music itself. Start an album, "
           "playlist or station in Music Assistant and ParlorWave Radio "
           "will follow it.")


class CatalogUnavailable(RuntimeError):
    """A catalog-dependent operation was asked for on an install that has
    no usable catalog. Carries the customer-facing sentence as its
    message so a caller can hand it straight to a person."""

    reason = REASON

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)


def usable(track_count: int) -> bool:
    """PURE. The verdict, as arithmetic, so the threshold is a table test."""
    return int(track_count or 0) >= MINIMUM_TRACKS


def report(count_fn: Callable[[], int] | None = None) -> dict[str, Any]:
    """The public capability fact. Never raises: an unreadable catalog is
    reported as an unusable one, because a surface that cannot be counted
    cannot be promised either."""
    try:
        count = int((count_fn or _count)() or 0)
    except Exception:  # noqa: BLE001 - a catalog read must not take health down
        logger.warning("catalog count unreadable; reporting unusable",
                       exc_info=True)
        count = 0
    return {"usable": usable(count), "track_count": count,
            "minimum_tracks": MINIMUM_TRACKS}


def require(count_fn: Callable[[], int] | None = None) -> None:
    """Raise CatalogUnavailable unless the catalog can really back an
    autonomous station. The gate behind every catalog-dependent service,
    so bypassing the interface fails with the same sentence the interface
    would have shown."""
    if not report(count_fn)["usable"]:
        raise CatalogUnavailable()


def _count() -> int:
    import library_repo
    return library_repo.count_tracks()
