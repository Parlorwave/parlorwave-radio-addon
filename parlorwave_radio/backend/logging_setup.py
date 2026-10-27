#!/usr/bin/env python3
"""The ONE place Music Brain's operational logging is configured.

WHY THIS EXISTS. Production runs `python -m uvicorn app:app --workers 1`.
The uvicorn CLI does configure logging -- it applies its own dictConfig -- but
that config declares handlers for exactly three loggers:

    uvicorn, uvicorn.error, uvicorn.access

and it has NO "root" key. So the root logger is left at its interpreter
defaults: no handlers, level WARNING. Every Music Brain module calls
logging.getLogger("musicbrain"), which is NOTSET, so it inherits that WARNING
from root and every logger.info() in the codebase is discarded at the level
check before a handler is ever consulted.

That also explains the one line operators DID see. A WARNING passes the level
check, propagates to a root with no handlers, and is emitted by
logging.lastResort -- a bare stderr handler that prints the bare message with
no timestamp, level or logger name. Music Brain was never "logging to stderr";
it was falling through to the standard library's emergency handler.

The cost is real: the automatic refill emits compact INFO decision and result
lines, and without a handler not one of them is visible; its behaviour could
then only be checked from SQLite and Music Assistant.

WHY NOT logging.basicConfig(). basicConfig configures the ROOT logger, which
would hand an INFO handler to every third-party logger in the process --
httpx, urllib3, asyncio and friends -- turning a fix for our own logs into a
dependency flood. We want ONE application logger at INFO, and everyone else
left exactly as they are (inheriting root at WARNING).

WHY A NAMED LOGGER IS SUFFICIENT. Every module in this package logs to the
single name "musicbrain". Configuring that one logger configures all of them,
which is why this file is small and why there is no per-module handler.

ORDERING IS SAFE IN BOTH DIRECTIONS. uvicorn builds its Config -- and applies
its dictConfig -- before it imports the application, so our call runs second
and wins. Even if that order ever changed, uvicorn's config sets
disable_existing_loggers=False and never names "musicbrain", so a later
dictConfig leaves this logger untouched.
"""

from __future__ import annotations

import logging
import os
import sys

# Every module in the package uses exactly this name.
LOGGER_NAME = "musicbrain"

ENV_VAR = "MUSICBRAIN_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"

# Compact and greppable: when, how bad, who, what. %(module)s earns its place
# because all twelve call sites share ONE logger name, so %(name)s alone could
# not tell station_refill from dj_engine. The timestamp matches the ISO-ish
# shape used for stations/refill_batches rows so log lines and database rows
# can be read side by side.
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s.%(module)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Stamped on the handler we own so configure() can recognise its own work and
# stay idempotent. Identity, not a name we ever match on by string equality
# from outside.
_OWNED = "_musicbrain_operational_handler"


def resolve_level(raw: str | None = None) -> int:
    """Map the configured level name to a logging level, failing safe.

    Anything unrecognised -- empty, misspelled, a number we do not accept --
    resolves to DEFAULT_LEVEL. A typo in an environment variable must never be
    able to silence production, which is the exact failure this module exists
    to fix."""
    if raw is None:
        raw = os.environ.get(ENV_VAR, "")
    name = str(raw).strip().upper()
    if not name:
        name = DEFAULT_LEVEL
    level = logging.getLevelName(name)
    # getLevelName returns the int for a known name, and the string
    # "Level %s" for anything else.
    if not isinstance(level, int):
        return logging.getLevelName(DEFAULT_LEVEL)
    return level


def configure(level: str | int | None = None, *, stream=None) -> logging.Logger:
    """Install Music Brain's single operational log handler. Idempotent.

    Safe to call more than once -- from app import, from a CLI entry point,
    from a test -- because it adopts the handler it already owns instead of
    stacking another one. Duplicate handlers mean duplicate lines, and a log
    that repeats itself is a log operators stop reading."""
    logger = logging.getLogger(LOGGER_NAME)

    if isinstance(level, int):
        resolved = level
    else:
        resolved = resolve_level(level)

    handler = None
    for existing in logger.handlers:
        if getattr(existing, _OWNED, False):
            handler = existing
            break

    if handler is None:
        handler = logging.StreamHandler(sys.stderr if stream is None else stream)
        setattr(handler, _OWNED, True)
        logger.addHandler(handler)
    elif stream is not None:
        # Re-pointing an existing owned handler (tests capture output this
        # way) must not create a second one.
        handler.setStream(stream)

    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.setLevel(logging.NOTSET)  # the logger owns the threshold
    logger.setLevel(resolved)

    # This logger has its own handler, so propagation could only ADD copies:
    # today via logging.lastResort, and permanently if anything ever attaches
    # a handler to root. Third-party loggers are untouched by this and keep
    # inheriting root at WARNING, which is what keeps dependency INFO quiet.
    logger.propagate = False

    return logger
