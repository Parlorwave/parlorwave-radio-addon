#!/usr/bin/env python3
"""Playback Intelligence: session state engine.

Owns sessions.db -- the ephemeral, TTL'd, crash-safe session store (third SQLite
file). Sessions record what played, what
the listener did, and the intended queue. Permanent behavioral history goes to
play_events in play_history.db (via musicdb); long-term taste stays behind the
existing feedback() path. This module never touches MusicBrainz, the
recommendation ledger, or Lidarr (enforced by an import-graph test).

Schema policy: sessions.db is operational state, not history. On a schema-version
mismatch the old file is archived with a timestamp (never silently deleted) and a
fresh DB is created.

CLI:
    python -m playback cleanup   # abandon idle sessions, purge expired ones
"""

import argparse
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import media_ref
import playback_arcs as arcs

from paths import BASE_DIR
import service_clock

BASE = BASE_DIR
# PLAYBACK_SESSIONS_DB overrides the location (migration/rollback is an env flip +
# file move; the default path benchmarks within targets, so the default
# stays).
SESSIONS_DB = Path(os.environ.get("PLAYBACK_SESSIONS_DB",
                                  str(BASE / "sessions.db")))
# Derived track features live in the knowledge layer (collection.db). The request
# path only READS it; the one writer is the `python -m playback features` job.
COLLECTION_DB = BASE / "collection.db"

SCHEMA_VERSION = 12  # v7: dj_breaks; v8: news_bulletins; v9: stations;
                     # v10: queue_ownership; v11: retired_program_tokens;
                     # v12: stations.program + stations.live_status (Mode 2)

# v8: the final playable news-bulletin ARTIFACT, regardless of origin.
# token is the internal id and future serving credential; publisher identity
# is UNIQUE(source_id, source_guid) -- GUIDs are scoped per source, and
# SQLite's multiple-NULL UNIQUE semantics exempt generated rows (NULL guid,
# NULL enclosure). produced_at is the origin-neutral downloaded/generated
# stamp. Owned by news_bulletins.py.
_NEWS_BULLETINS_DDL = """CREATE TABLE IF NOT EXISTS news_bulletins (
        token TEXT PRIMARY KEY,
        origin_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        source_guid TEXT,
        title TEXT DEFAULT '',
        published_at TEXT,
        feed_url TEXT DEFAULT '',
        enclosure_url TEXT,
        content_type TEXT DEFAULT '',
        expected_length INTEGER,
        audio_path TEXT DEFAULT '',
        size_bytes INTEGER,
        duration_s REAL,
        produced_at TEXT,
        state TEXT NOT NULL DEFAULT 'discovered',
        error TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(source_id, source_guid)
    )"""
#                  + sessions.last_progress_at (playback advanced, not just events)
#     v5: session_tracks media identity (provider items without a local path)
#     v6: sessions.paused_at + pause_position/pause_duration (pause grace:
#         a paused attachment survives restarts and closes neutrally at its
#         frozen checkpoint when the customer grace expires)
#     v7: dj_breaks table (queue-native DJ break lifecycle; restart-safe,
#         duplicate-insertion-proof)

BUSY_TIMEOUT_MS = int(os.environ.get("MUSICBRAIN_SQLITE_BUSY_TIMEOUT_MS", "5000"))

# Skip/complete inference thresholds.
SKIP_RATIO = float(os.environ.get("PLAYBACK_SKIP_RATIO", "0.40"))
COMPLETE_RATIO = float(os.environ.get("PLAYBACK_COMPLETE_RATIO", "0.80"))
# Identical raw events within this window are automation double-fires.
DEDUP_WINDOW_SECONDS = float(os.environ.get("PLAYBACK_DEDUP_WINDOW_S", "5"))
IDLE_ABANDON_MINUTES = float(os.environ.get("PLAYBACK_IDLE_ABANDON_MIN", "20"))
SESSION_TTL_DAYS = float(os.environ.get("PLAYBACK_SESSION_TTL_DAYS", "30"))
# A session is a claim about the present. Age alone is NOT a reason to
# end one. SESSION_MAX_AGE_HOURS is the age past which a session must PROVE it
# is still alive; a session that keeps proving it stays active indefinitely.
#
# Why not an unconditional expiry keyed on started_at: that guards against a
# zombie session -- a stale event source (an HA helper holding a session id)
# feeding events to a session id that is already dead. The lifecycle routes by
# player and v4 routes by queue_id, so an event can only originate from a real
# player. An unconditional expiry would instead kill genuinely active sessions
# mid-track, so liveness is required.
SESSION_MAX_AGE_HOURS = float(os.environ.get("PLAYBACK_SESSION_MAX_AGE_H", "12"))
# An old session renews its lease by RECEIVING events...
SESSION_LEASE_MINUTES = float(os.environ.get("PLAYBACK_SESSION_LEASE_MIN", "20"))
# ...and by MAKING PROGRESS. Events alone are not liveness: a wedged player
# repeating one track forever still emits position events on a timer.
#
# Progress is recorded in last_progress_at, which advances ONLY on evidence
# that playback actually moved: a track resolving, a new track starting, or a
# position report that DIFFERS from the last one. Note what is excluded --
# extrapolated position (raw checkpoint + wall clock) would keep climbing on a
# frozen player and manufacture progress out of nothing.
#
# The deadline scales with the media actually playing, because the raw HA
# checkpoint is coarse: a player may report only a couple of positions per
# track (the start and one mid-track), with minutes of silence between
# them. A three-hour track can therefore legitimately show no new information
# for a very long time. Scaling by duration means a 4-minute song must show
# progress within the floor, while long-form media gets proportionate room.
SESSION_PROGRESS_MIN_HOURS = float(os.environ.get("PLAYBACK_PROGRESS_MIN_H", "2"))
SESSION_PROGRESS_MAX_HOURS = float(os.environ.get("PLAYBACK_PROGRESS_MAX_H", "12"))
PROGRESS_DURATION_FACTOR = float(os.environ.get("PLAYBACK_PROGRESS_FACTOR", "1.5"))

# Fallback grace when the settings document is unreadable (mirrors the
# settings default; settings must never take lifecycle down).
DEFAULT_RESUME_GRACE_MINUTES = 30.0
# resume_grace_minutes = null means NEVER: a genuinely paused attachment is
# not ended because time passed -- there is deliberately no hidden age
# ceiling behind the customer's "Never". A paused session still closes on
# POSITIVE lifecycle evidence (explicit stop/off, queue replacement,
# supersede, a new attachment on the queue). Honest limitation: with today's
# inputs the backend cannot distinguish "still paused after N days" from
# "player vanished N days ago", so with Never selected a vanished player's
# paused session persists until real evidence arrives (e.g. future HA
# 'unavailable' forwarding, or the next playback on that queue superseding
# it). That persistence is the documented cost of Never, not a bug.

# Lifecycle: 'superseded' = replaced on its player by a newer session;
# 'expired' = outlived SESSION_MAX_AGE_HOURS *without* proving liveness.
VALID_SESSION_STATUS = {"active", "ended", "abandoned", "superseded", "expired"}
_END_STATUSES = ("ended", "abandoned", "superseded", "expired")
VALID_TRACK_OUTCOMES = {"pending", "played", "skipped", "partial", "removed"}


def _iso(now: datetime | None = None) -> str:
    return (now or service_clock.legacy_now()).isoformat(timespec="seconds")


def _db(db_path: Path | None = None) -> Path:
    """Resolve the sessions DB path at call time (so tests can monkeypatch
    playback.SESSIONS_DB; a def-time default would bind the original constant)."""
    return Path(db_path) if db_path is not None else Path(SESSIONS_DB)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    c = sqlite3.connect(_db(db_path))
    c.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    return c


def _schema_version(c: sqlite3.Connection) -> int | None:
    try:
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row else None


# --- v9: durable station identity (station replenishment) --------------------
#
# A STATION is a first-class product object, deliberately distinct from a
# playback SESSION and from an output QUEUE. Why: one queue can carry several
# different session_ids over a few weeks, and
# playback.resume_grace_minutes (e.g. 30) abandons a paused session
# overnight -- so a station keyed on either would die for reasons that have
# nothing to do with the station. A station lives potentially indefinitely:
# it ends when it is retired as a product object, never on a clock.
#
# Three lifecycle values and no more:
#
#   preparing    a WRITE-AHEAD INTENT record. Music Assistant cannot join a
#                SQLite transaction, so the row is committed BEFORE the
#                replace_future call and carries the compact witness that
#                lets restart prove whether that external write landed. The
#                OLD station stays 'active' throughout, so a failed rebuild
#                leaves the listener exactly where they were.
#   active       bound to a queue and eligible for replenishment.
#   deactivated  terminal. Superseded, replaced, cleared, or GC'd.
#
# There is deliberately NO 'suspended'. Paused, missing player and "the user
# queued 100 tracks ahead of us" are transient conditions of the world, not
# changes to what the station IS; they are derived at read time as a
# refill_blocked_reason. Making them durable would churn the row on every
# pause and let a momentary blip become permanent state.
#
# There is deliberately NO `generation` column. Every explicit Start Radio
# mints a NEW station_id, there is no in-place profile mutation and no
# rebinding, so station_id + status + the partial unique indexes already fence
# every race a generation counter was proposed for -- it would sit at 1
# forever. It can arrive later as one additive ALTER TABLE if a real
# same-station mutation is ever introduced.
#
# NOTHING here derives from MA's `sort_index`. It is strictly increasing
# but NOT contiguous (an option:"add" can produce ...499, 501,
# 502), so positions come from ordered-list order and identity from
# queue_item_id -- never from arithmetic.
_STATIONS_DDL = """CREATE TABLE IF NOT EXISTS stations (
        station_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'preparing'
            CHECK (status IN ('preparing', 'active', 'deactivated')),
        -- frozen SEMANTIC identity: what this station IS. Selection policy
        -- and taste stay live, so a station started today is not pinned to
        -- today's internal knobs.
        recipe_version INTEGER NOT NULL DEFAULT 1,
        request_text TEXT NOT NULL DEFAULT '',
        seed_artist TEXT NOT NULL DEFAULT '',
        seed_title TEXT NOT NULL DEFAULT '',
        seed_media_key TEXT NOT NULL DEFAULT '',
        profile_source TEXT NOT NULL DEFAULT 'station',
        -- The frozen LLM output. build_station() already returns this dict and
        -- station_radio currently discards it; persisting it is what makes
        -- every later refill deterministic with NO Ollama call.
        artist_weights TEXT NOT NULL DEFAULT '{}',
        -- current output binding, set once at activation
        queue_id TEXT NOT NULL DEFAULT '',
        -- preparing-recovery witness: enough to prove whether replace_future
        -- landed, without storing a queue dump
        intended_uris TEXT NOT NULL DEFAULT '[]',
        pre_current_queue_item_id TEXT NOT NULL DEFAULT '',
        pre_future_music_count INTEGER NOT NULL DEFAULT -1,
        pre_future_music_hash TEXT NOT NULL DEFAULT '',
        prepared_at TEXT,
        activated_at TEXT,
        deactivated_at TEXT,
        deactivated_reason TEXT NOT NULL DEFAULT '',
        -- caches: observability, cooldown and GC inputs. NEVER a functional
        -- expiry -- a station paused overnight is the same station.
        last_seen_at TEXT,
        last_replenished_at TEXT,
        refill_count INTEGER NOT NULL DEFAULT 0,
        -- bounded ownership witness: [{queue_item_id, uri, last_region}].
        -- Ownership evidence only; play history lives in session_tracks and
        -- play_events. tail_is_partial=1 means owned runway is a LOWER BOUND.
        tail TEXT NOT NULL DEFAULT '[]',
        tail_is_partial INTEGER NOT NULL DEFAULT 0,
        -- bounded rotation ring: [{uri, at}]
        generated TEXT NOT NULL DEFAULT '[]',
        -- v12 (Station Programmer, Mode 2): the frozen program and the
        -- latest diagnosis. '{}' on a legacy Start Radio station.
        program TEXT NOT NULL DEFAULT '{}',
        live_status TEXT NOT NULL DEFAULT '{}'
    )"""

# One refill attempt. player_queues/play_media answers HTTP 200
# with body `null` -- no inserted ids, no echo -- so a queue READBACK is the
# only reconciliation evidence that will ever exist, and it can only be read
# against a witness captured BEFORE the write. Hence these columns.
_REFILL_BATCHES_DDL = """CREATE TABLE IF NOT EXISTS refill_batches (
        batch_id TEXT PRIMARY KEY,
        station_id TEXT NOT NULL,
        queue_id TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending'
            CHECK (state IN ('pending', 'confirmed', 'not_landed',
                             'abandoned')),
        intended_uris TEXT NOT NULL,
        -- from player_queues/all.items, which has NO 500-item cap (unlike
        -- player_queues/items, whose default limit silently truncates)
        pre_items_count INTEGER NOT NULL,
        pre_tail_queue_item_id TEXT NOT NULL DEFAULT '',
        -- ORDERED-LIST POSITION, never a sort_index
        pre_tail_position INTEGER NOT NULL DEFAULT -1,
        pre_qiid_hash TEXT NOT NULL DEFAULT '',
        pre_uri_multiplicity TEXT NOT NULL DEFAULT '{}',
        -- diagnostics only. A timer may TRIGGER reconciliation; it may never
        -- decide an outcome, because an append whose response was lost is
        -- indistinguishable from one that never happened until MA is read.
        claimed_at TEXT NOT NULL,
        resolved_at TEXT,
        resolution TEXT NOT NULL DEFAULT '',
        attempts INTEGER NOT NULL DEFAULT 0
    )"""

# v1 is explicitly single-station, and the rest of the engine already assumes
# it: playback.start_session supersedes to keep one active session per player,
# ma_observer.resolve_target refuses to guess with more than one active session
# (targeting nothing, which would switch replenishment off entirely), and
# station_radio refuses active_station_elsewhere. Two SEPARATE partial indexes
# on two SEPARATE predicates, because the Start Radio choreography REQUIRES one
# 'active' and one 'preparing' to coexist. Deactivated rows are unconstrained,
# so history accumulates freely until GC.
_STATIONS_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_stations_one_active "
    "ON stations(status) WHERE status = 'active'",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_stations_one_preparing "
    "ON stations(status) WHERE status = 'preparing'",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_refill_one_pending "
    "ON refill_batches(station_id) WHERE state = 'pending'",
    "CREATE INDEX IF NOT EXISTS ix_refill_station "
    "ON refill_batches(station_id, claimed_at)",
]

_V9_DDL = [_STATIONS_DDL, _REFILL_BATCHES_DDL, *_STATIONS_INDEXES]

# Queue ownership: the record of ParlorWave having taken over
# MUSIC PROGRAMMING for the queue behind an active station, and of what was
# done to Music Assistant's per-queue Autoplay flag to make that true. One
# row per queue, rewritten on every acquisition, so the table answers "who
# programs this queue right now and what did MA say before we touched it".
# Durable rather than in-memory so the drift budget and the pre-ownership
# observation survive a Music Brain restart -- the flag itself lives in MA.
_QUEUE_OWNERSHIP_DDL = """CREATE TABLE IF NOT EXISTS queue_ownership (
        queue_id TEXT PRIMARY KEY,
        station_id TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'unverified'
            CHECK (state IN ('unverified', 'owned', 'contested', 'released')),
        -- what MA reported BEFORE Parlorwave touched the flag: 1 / 0 / NULL
        -- (NULL = could not be read). Recorded for the audit trail and for a
        -- future restore contract; nothing reads it back to act on it.
        autoplay_before INTEGER,
        autoplay_verified_at TEXT,
        disable_writes INTEGER NOT NULL DEFAULT 0,
        drift_corrections INTEGER NOT NULL DEFAULT 0,
        drift_window_started_at TEXT,
        last_checked_at TEXT,
        last_error TEXT NOT NULL DEFAULT '',
        acquired_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        released_at TEXT,
        released_reason TEXT NOT NULL DEFAULT ''
    )"""

_V10_DDL = [_QUEUE_OWNERSHIP_DDL]


#: v11. THE ONE THING A RETIRED PROGRAM TOKEN HAS TO OUTLIVE ITS OWN ROW FOR.
#:
#: A Music Assistant queue item is addressable forever; the rows that describe
#: our program audio are deliberately bounded (dj_breaks is reaped at 24h,
#: news_bulletins at ROW_CAP). Those two lifetimes disagree, and the gap is
#: not theoretical: aired DJ tokens whose rows had been reaped answer 404
#: to a re-exposed queue item -- the exact transport-fatal
#: outcome program_audio.py exists to prevent.
#:
#: So retirement becomes a FACT that survives the row. Deliberately the
#: smallest possible identity: which token, what kind of program audio it was,
#: and when it was retired. No spoken text, no paths, no telemetry, no
#: customer content -- this table answers one question, "has this token been
#: retired", and must never become a second history.
_RETIRED_TOKENS_DDL = """CREATE TABLE IF NOT EXISTS retired_program_tokens (
        token TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        retired_at TEXT NOT NULL
    )"""
_RETIRED_TOKENS_INDEX = ("CREATE INDEX IF NOT EXISTS retired_program_tokens_at "
                         "ON retired_program_tokens (retired_at)")
_V11_DDL = [_RETIRED_TOKENS_DDL, _RETIRED_TOKENS_INDEX]


#: v12. THE STATION PROGRAMMER'S TWO COLUMNS (Parlorwave Mode 2).
#:
#: A Mode 2 station is an ordinary `stations` row -- the same write-ahead
#: `preparing` record, the same one-active invariant, the same ownership
#: witness and rotation ring, the same refill batches -- plus two small JSON
#: documents the v9 row had nowhere to put:
#:
#:   program      what the station IS and where it plays, frozen at start:
#:                the StationIntent (kind, label, query, preset id, origin),
#:                the target output, the draw it opened with, the live-
#:                viability verdict, and the queue items that were already
#:                ahead when it started (so a later listener addition can
#:                be told apart from something that was there before).
#:   live_status  the latest diagnosis, overwritten in place: the last
#:                runway check and the last refill result. Observability
#:                only -- nothing decides from it.
#:
#: A fresh install gets both columns from _STATIONS_DDL. The upgrade step
#: adds whichever is MISSING, so it is idempotent: a database that reaches
#: step 9 through the current DDL (a v8 install, or a store whose tables
#: were built from _DDL) already has them, and one at v11 gains them here.
#: Legacy Start Radio rows read '{}' in both, which is what they have always
#: meant.
_V12_COLUMNS = (("program", "TEXT NOT NULL DEFAULT '{}'"),
                ("live_status", "TEXT NOT NULL DEFAULT '{}'"))


def _v12_station_columns(c) -> None:
    have = {row[1] for row in c.execute("PRAGMA table_info(stations)")}
    for name, decl in _V12_COLUMNS:
        if name not in have:
            c.execute(f"ALTER TABLE stations ADD COLUMN {name} {decl}")


_V12_DDL = [_v12_station_columns]


_DDL = [
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    _RETIRED_TOKENS_DDL, _RETIRED_TOKENS_INDEX,
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        player TEXT DEFAULT '',
        request TEXT DEFAULT '',
        mode TEXT DEFAULT 'default_adaptive',
        arc_stage TEXT,
        arc_started_at TEXT,
        target_energy REAL,
        observed_energy REAL,
        queue_depth INTEGER DEFAULT 0,
        current_seq INTEGER,
        current_path TEXT,
        last_position REAL,
        last_duration REAL,
        last_event_at TEXT,
        last_banter_at TEXT,
        context TEXT DEFAULT '{}',
        seed INTEGER,
        status TEXT NOT NULL DEFAULT 'active',
        origin TEXT NOT NULL DEFAULT 'requested',
        queue_owned INTEGER NOT NULL DEFAULT 1,
        queue_id TEXT NOT NULL DEFAULT '',
        last_progress_at TEXT,
        current_media_key TEXT NOT NULL DEFAULT '',
        -- v6 pause grace. paused_at NULL = not paused. pause_position/duration
        -- freeze the Step-A2 checkpoint at the moment of pause; every close
        -- of a paused session resolves at THIS position, never at whatever
        -- stale value later events might carry.
        paused_at TEXT,
        pause_position REAL,
        pause_duration REAL
    )""",
    """CREATE TABLE IF NOT EXISTS session_tracks (
        session_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        -- The CATALOG path, '' when this item is not a local library track.
        -- Identity lives in media_key; this column answers "which library row",
        -- not "what is playing". It stays NOT NULL so every existing row and
        -- query is untouched, and '' is the schema's existing convention for
        -- absent text -- it is not a fabricated filesystem path.
        path TEXT NOT NULL,
        artist TEXT DEFAULT '',
        album TEXT DEFAULT '',
        title TEXT DEFAULT '',
        genre TEXT DEFAULT '',
        year TEXT DEFAULT '',
        queued_at TEXT,
        started_at TEXT,
        resolved_at TEXT,
        outcome TEXT NOT NULL DEFAULT 'pending',
        resolve_reason TEXT,
        energy REAL,
        score REAL,
        score_reason TEXT,
        source TEXT DEFAULT 'station',
        -- v5 media identity. media_key is the stable source identifier
        -- (provider URI, else file://<catalog path>); '' means the player
        -- reported none, so the item exists only within this session.
        media_key TEXT NOT NULL DEFAULT '',
        media_uri TEXT NOT NULL DEFAULT '',
        -- IDENTITY namespace (URI scheme), not the content provider: HA never
        -- exposes the latter. See media_ref.namespace_of.
        media_namespace TEXT NOT NULL DEFAULT '',
        -- RESOLUTION state: catalog | unresolved. Never provenance.
        match_status TEXT NOT NULL DEFAULT 'catalog',
        PRIMARY KEY (session_id, seq)
    )""",
    """CREATE TABLE IF NOT EXISTS session_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        at TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload TEXT DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS dj_breaks (
        -- Queue-native DJ break lifecycle (v7). One row per break intent;
        -- the token is the audio identity under /broadcast/audio/.
        token TEXT PRIMARY KEY,
        queue_id TEXT NOT NULL,
        session_id TEXT DEFAULT '',
        state TEXT NOT NULL DEFAULT 'planned',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        spoken_text TEXT DEFAULT '',
        audio_path TEXT DEFAULT '',
        duration_s REAL,
        telemetry TEXT DEFAULT '{}',
        error TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS session_constraints (
        session_id TEXT NOT NULL,
        constraint_id TEXT NOT NULL,
        kind TEXT, field TEXT, value TEXT, polarity TEXT,
        strength TEXT, scope TEXT, ttl TEXT, created_at TEXT,
        PRIMARY KEY (session_id, constraint_id)
    )""",
    _NEWS_BULLETINS_DDL,
    _STATIONS_DDL,
    _REFILL_BATCHES_DDL,
    *_STATIONS_INDEXES,
    _QUEUE_OWNERSHIP_DDL,
    "CREATE INDEX IF NOT EXISTS idx_session_events_sid ON session_events(session_id, at)",
    "CREATE INDEX IF NOT EXISTS idx_session_tracks_path ON session_tracks(session_id, path)",
]



def _archive_path(path: Path, version: int | None, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.name}.v{version or 0}.{stamp}.bak")


# In-place upgrades, keyed by the version they PRODUCE. A version is migratable
# only if every step from the stored version up to SCHEMA_VERSION is listed here;
# anything else still falls back to archive-and-recreate. Additive DDL only --
# a migration must never be able to lose a row.
_MIGRATIONS: dict[int, list[str]] = {
    # One version, two columns: both land before v4 ever deploys, so there is
    # no reason to make an installation migrate twice.
    4: ["ALTER TABLE sessions ADD COLUMN queue_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE sessions ADD COLUMN last_progress_at TEXT"],
    # v5 is additive only -- no table rebuild. Every existing session_tracks
    # row has a real local path, so 'catalog' is the correct default for them
    # and their meaning is preserved exactly.
    5: ["ALTER TABLE session_tracks ADD COLUMN media_key TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE session_tracks ADD COLUMN media_uri TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE session_tracks ADD COLUMN media_namespace TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE session_tracks ADD COLUMN match_status TEXT NOT NULL "
        "DEFAULT 'catalog'",
        "ALTER TABLE sessions ADD COLUMN current_media_key TEXT NOT NULL DEFAULT ''"],
    # v6 pause grace: additive, NULL = not paused, so every existing session
    # is correctly "not paused" the moment the migration lands.
    6: ["ALTER TABLE sessions ADD COLUMN paused_at TEXT",
        "ALTER TABLE sessions ADD COLUMN pause_position REAL",
        "ALTER TABLE sessions ADD COLUMN pause_duration REAL"],
    # v7 queue-native DJ: one new table, nothing existing touched.
    7: ["""CREATE TABLE IF NOT EXISTS dj_breaks (
        token TEXT PRIMARY KEY,
        queue_id TEXT NOT NULL,
        session_id TEXT DEFAULT '',
        state TEXT NOT NULL DEFAULT 'planned',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        spoken_text TEXT DEFAULT '',
        audio_path TEXT DEFAULT '',
        duration_s REAL,
        telemetry TEXT DEFAULT '{}',
        error TEXT DEFAULT ''
    )"""],
    # v8 is one additive table -- no existing table changes, no data touched.
    8: [_NEWS_BULLETINS_DDL],
    # v9: two new tables plus their indexes. Nothing existing is read,
    # rewritten or dropped -- sessions, session_tracks, session_events,
    # dj_breaks and news_bulletins are untouched by this step.
    9: _V9_DDL,
    # v10: one additive table (queue_ownership). Nothing existing is read,
    # rewritten or dropped; stations and refill_batches keep their shape.
    10: _V10_DDL,
    # v11: one additive table plus its index. Nothing existing is read,
    # rewritten or dropped -- dj_breaks and news_bulletins keep their shape
    # and their contents. An installation that migrates has an EMPTY ledger,
    # which is the correct starting point: every token it already knows about
    # still has its row, and rows are consulted first.
    11: _V11_DDL,
    # v12: two additive columns on stations (the Station Programmer's frozen
    # program and its latest diagnosis), added only where missing. No row is
    # read or rewritten; every existing station reads '{}' for both, which
    # is what a legacy Start Radio station has always meant.
    12: _V12_DDL,
}


def _migration_path(from_version: int | None) -> list[int] | None:
    """The ordered versions to apply, or None when no in-place path exists."""
    if from_version is None or from_version >= SCHEMA_VERSION:
        return None
    steps = list(range(from_version + 1, SCHEMA_VERSION + 1))
    return steps if all(v in _MIGRATIONS for v in steps) else None


def _backup_path(path: Path, version: int | None, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.name}.pre-v{SCHEMA_VERSION}.from-v{version or 0}.{stamp}.bak")


def _migrate_in_place(path: Path, version: int, steps: list[int],
                      now: datetime | None = None) -> Path:
    """Upgrade sessions.db without destroying it. Copies a safety backup first,
    applies every step inside ONE transaction, and restores the backup if any
    step fails -- a half-upgraded database is not a reachable state.
    Returns the backup path."""
    backup = _backup_path(path, version, now)
    # Fold any WAL back into the main file first, so the backup is a complete,
    # self-contained database rather than a torso that needs its sidecars.
    c = _connect(path)
    try:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        c.close()
    shutil.copy2(path, backup)

    c = _connect(path)
    try:
        c.execute("BEGIN IMMEDIATE")
        for target in steps:
            for ddl in _MIGRATIONS[target]:
                # A step is SQL, or a function of the open connection for
                # a change SQLite cannot express idempotently (v12's
                # add-column-if-missing). Same transaction either way.
                if callable(ddl):
                    ddl(c)
                else:
                    c.execute(ddl)
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                  (str(SCHEMA_VERSION),))
        c.commit()
    except Exception:
        try:
            c.rollback()
        finally:
            c.close()
        # SQLite DDL is transactional, so the rollback alone is usually enough.
        # Restore anyway, and drop the sidecars with it: replacing the main file
        # while a stale -wal survives would let SQLite replay post-restore frames
        # over the recovered database.
        shutil.copy2(backup, path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        raise
    finally:
        try:
            c.close()
        except Exception:
            pass
    return backup


def ensure_sessions_schema(db_path: Path | None = None, now: datetime | None = None) -> Path | None:
    """Create/verify sessions.db.

    A version gap with a known in-place upgrade path is MIGRATED (safety backup
    first, one transaction, restore on failure) so sessions and history survive.
    Only an unknown/unreachable version falls back to archive-and-recreate.
    Returns the backup/archive path when either happened, else None."""
    path = _db(db_path)
    archived = None
    if path.exists():
        c = _connect(path)
        try:
            version = _schema_version(c)
        finally:
            c.close()
        if version == SCHEMA_VERSION:
            return None
        steps = _migration_path(version)
        if steps:
            return _migrate_in_place(path, version, steps, now)
        archived = _archive_path(path, version, now)
        path.rename(archived)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sidecar.rename(Path(str(archived) + suffix))
    c = _connect(path)
    try:
        for ddl in _DDL:
            c.execute(ddl)
        c.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()
    finally:
        c.close()
    return archived


# --- Session lifecycle -------------------------------------------------------

_SESSION_COLS = [
    "session_id", "started_at", "ended_at", "player", "request", "mode",
    "arc_stage", "arc_started_at", "target_energy", "observed_energy",
    "queue_depth", "current_seq", "current_path", "last_position",
    "last_duration", "last_event_at", "last_banter_at", "context", "seed",
    "status", "origin", "queue_owned", "queue_id", "last_progress_at",
    "current_media_key", "paused_at", "pause_position", "pause_duration",
]


def _session_dict(row: tuple) -> dict[str, Any]:
    d = dict(zip(_SESSION_COLS, row))
    try:
        d["context"] = json.loads(d.get("context") or "{}")
    except (TypeError, ValueError):
        d["context"] = {}
    d["queue_owned"] = bool(d.get("queue_owned", 1))
    return d


def start_session(request: str = "", mode: str = "default_adaptive", player: str = "",
                  context: dict[str, Any] | None = None, seed: int | None = None,
                  now: datetime | None = None, origin: str = "requested",
                  queue_owned: bool = True, queue_id: str = "",
                  db_path: Path | None = None) -> dict[str, Any]:
    """Create an active session and return its state dict.

    At most one active session per player. Starting (or adopting) a
    session on a player that already has one auto-ends the old session as
    'superseded' -- the stale-helper failure mode cannot produce two truths
    for one speaker. origin ('requested'|'adopted') is provenance metadata;
    queue_owned is the behavioral switch (False: the backend did not build
    this session's queue, so it never promises the next track)."""
    if origin not in ("requested", "adopted"):
        raise ValueError(f"invalid session origin: {origin}")
    ensure_sessions_schema(db_path)
    session_id = uuid.uuid4().hex
    at = _iso(now)
    now_dt = now or service_clock.legacy_now()
    if seed is None:
        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
    superseded = _supersede_sessions(player, queue_id, session_id, now_dt, db_path)
    # Arc initialization: the session starts in its template's first stage.
    # The stored mode string is preserved verbatim (a stored contract); an unknown
    # mode runs the Default Adaptive template (arcs.template_for fallback).
    mode_key = mode or "default_adaptive"
    stage = arcs.initial_stage(mode_key)
    start_payload = {"request": request, "mode": mode_key,
                     "arc_stage": stage["key"], "player": player,
                     "origin": origin}
    if queue_id:
        start_payload["queue_id"] = queue_id
    if superseded:
        start_payload["superseded"] = superseded
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO sessions (session_id, started_at, player, request, mode, "
            "arc_stage, arc_started_at, target_energy, "
            "context, seed, last_event_at, last_progress_at, status, origin, "
            "queue_owned, queue_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
            (session_id, at, player or "", request or "", mode_key,
             stage["key"], at, arcs.target_energy_for(stage, None),
             json.dumps(context or {}, ensure_ascii=False), seed, at, at,
             origin, 1 if queue_owned else 0, queue_id or ""),
        )
        c.execute(
            "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, 'start', ?)",
            (session_id, at, json.dumps(start_payload, ensure_ascii=False)),
        )
        c.commit()
        row = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        c.close()
    return _session_dict(row)


def get_session(session_id: str, recent: int = 10,
                db_path: Path | None = None) -> dict[str, Any] | None:
    """Session state + the recent resolved tail + pending queue. None if unknown."""
    ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        session = _session_dict(row)
        resolved = c.execute(
            "SELECT seq, path, artist, title, album, outcome, started_at, resolved_at, source "
            "FROM session_tracks WHERE session_id = ? AND outcome != 'pending' "
            "ORDER BY seq DESC LIMIT ?", (session_id, max(0, recent)),
        ).fetchall()
        pending = c.execute(
            "SELECT seq, path, artist, title, album, source "
            "FROM session_tracks WHERE session_id = ? AND outcome = 'pending' AND started_at IS NULL "
            "ORDER BY seq", (session_id,),
        ).fetchall()
    finally:
        c.close()
    session["recent"] = [
        {"seq": r[0], "path": r[1], "artist": r[2], "title": r[3], "album": r[4],
         "outcome": r[5], "started_at": r[6], "resolved_at": r[7], "source": r[8]}
        for r in reversed(resolved)
    ]
    session["queue_pending"] = [
        {"seq": r[0], "path": r[1], "artist": r[2], "title": r[3], "album": r[4],
         "source": r[5]}
        for r in pending
    ]
    return session


def resolved_session_tracks(session_id: str, limit: int = 200,
                            db_path: Path | None = None) -> list[dict[str, Any]]:
    """The NEWEST `limit` resolved tracks of a session, in play order,
    including year and resolve_reason (fields get_session's recent tail
    omits). Tail selection matters: in a long-running session a head-first
    LIMIT would freeze the window at the first rows forever (a
    stale-callback bug). The DJ break renderer is the live
    consumer."""
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT artist, title, year, outcome, source, resolved_at, started_at, "
            "resolve_reason "
            "FROM session_tracks WHERE session_id = ? AND outcome IN "
            "('played','partial','skipped') ORDER BY seq DESC LIMIT ?",
            (session_id, limit)).fetchall()
    finally:
        c.close()
    cols = ["artist", "title", "year", "outcome", "source", "resolved_at",
            "started_at", "resolve_reason"]
    return [dict(zip(cols, r)) for r in reversed(rows)]


def end_session(session_id: str, status: str = "ended", now: datetime | None = None,
                db_path: Path | None = None) -> bool:
    """Mark a session ended/abandoned/superseded/expired. Returns False if
    unknown or not active."""
    if status not in _END_STATUSES:
        raise ValueError(f"invalid end status: {status}")
    ensure_sessions_schema(db_path)
    at = _iso(now)
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE sessions SET status = ?, ended_at = ?, last_event_at = ? "
            "WHERE session_id = ? AND status = 'active'",
            (status, at, at, session_id),
        )
        if cur.rowcount == 0:
            return False
        c.execute(
            "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, 'end', ?)",
            (session_id, at, json.dumps({"status": status})),
        )
        c.commit()
    finally:
        c.close()
    return True


def _supersede_sessions(player: str, queue_id: str, new_session_id: str,
                        now_dt: datetime,
                        db_path: Path | None = None) -> list[str]:
    """End every active session for this OUTPUT as 'superseded':
    resolve any in-flight track through the ordinary inference ladder, then
    record a cross-referencing 'superseded' event naming the replacement.
    Returns the superseded session ids.

    Identity: when a queue_id is known it is authoritative, because one MA
    queue is one listening session no matter how many player entities report
    it (a group and each of its children all report the group's queue_id).
    Falling back to `player` keeps pre-queue_id callers behaving exactly as
    before."""
    if queue_id:
        where, args = "queue_id = ? AND status = 'active'", (queue_id,)
    elif player:
        where, args = "player = ? AND status = 'active'", (player,)
    else:
        return []
    c = _connect(db_path)
    try:
        rows = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions WHERE {where}", args,
        ).fetchall()
        sessions = [_session_dict(r) for r in rows]
    finally:
        c.close()
    superseded = []
    for session in sessions:
        c = _connect(db_path)
        try:
            _resolve_current(c, session, ratio=_effective_ratio(session),
                             explicit_skip=False, source="inferred",
                             reason="queue_replacement", now=now_dt)
            c.commit()
        finally:
            c.close()
        if end_session(session["session_id"], status="superseded", now=now_dt,
                       db_path=db_path):
            append_event(session["session_id"], "superseded",
                         {"by": new_session_id}, now=now_dt, db_path=db_path,
                         dedup=False)
            superseded.append(session["session_id"])
    return superseded


def _supersede_player_sessions(player: str, new_session_id: str,
                               now_dt: datetime,
                               db_path: Path | None = None) -> list[str]:
    """Back-compat shim: supersede by player entity only."""
    return _supersede_sessions(player, "", new_session_id, now_dt, db_path)


def attach_queue_id(session_id: str, queue_id: str,
                    db_path: Path | None = None) -> bool:
    """Back-fill a queue_id onto a session that predates queue identity (or was
    created by a caller that didn't know it yet). Never overwrites a queue_id
    that is already set -- an established session's identity is immutable."""
    if not session_id or not queue_id:
        return False
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE sessions SET queue_id = ? "
            "WHERE session_id = ? AND (queue_id IS NULL OR queue_id = '')",
            (queue_id, session_id))
        c.commit()
        return cur.rowcount > 0
    finally:
        c.close()


def get_active_for_queue(queue_id: str,
                         db_path: Path | None = None) -> dict[str, Any] | None:
    """The active session for a Music Assistant queue, or None.

    This is the deduplication primitive: a group and every child that reports
    the group's queue_id all resolve to the SAME session, so one physical
    playback can never become several sessions (and, later, never several
    inserted breaks)."""
    if not queue_id:
        return None
    ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions "
            "WHERE queue_id = ? AND status = 'active' "
            "ORDER BY started_at DESC LIMIT 1", (queue_id,),
        ).fetchone()
    finally:
        c.close()
    return _session_dict(row) if row else None


def record_queue(session_id: str, tracks: list[dict[str, Any]], source: str = "station",
                 now: datetime | None = None, db_path: Path | None = None) -> int:
    """Append tracks to the session's intended queue as pending rows (denormalized
    at queue time). Returns the number of rows added."""
    ensure_sessions_schema(db_path)
    at = _iso(now)
    c = _connect(db_path)
    try:
        row = c.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM session_tracks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row[0])
        added = 0
        for t in tracks:
            path = str(t.get("path") or "").strip()
            if not path:
                continue
            seq += 1
            c.execute(
                "INSERT INTO session_tracks (session_id, seq, path, artist, album, title, "
                "genre, year, queued_at, outcome, source, energy, score, score_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (session_id, seq, path, t.get("artist") or "", t.get("album") or "",
                 t.get("title") or "", t.get("genre") or "", str(t.get("year") or ""),
                 at, t.get("source") or source,   # per-track override (a lane's own source)
                 t.get("energy"), t.get("score"), t.get("score_reason")),
            )
            added += 1
        if added:
            c.execute(
                "UPDATE sessions SET queue_depth = "
                "(SELECT COUNT(*) FROM session_tracks WHERE session_id = ? AND outcome = 'pending' AND started_at IS NULL), "
                "last_event_at = ? WHERE session_id = ?",
                (session_id, at, session_id),
            )
            c.execute(
                "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, 'refill', ?)",
                (session_id, at, json.dumps({"added": added, "source": source})),
            )
        c.commit()
    finally:
        c.close()
    return added


# --- Raw event log with double-fire dedup ------------------------------------

def _dedup_key(payload: dict[str, Any] | None) -> str:
    p = payload or {}
    return str(p.get("path") or p.get("media_title") or "")


def append_event(session_id: str, kind: str, payload: dict[str, Any] | None = None,
                 now: datetime | None = None, db_path: Path | None = None,
                 dedup: bool = True) -> bool:
    """Append a raw session event. Identical events (same session, kind, and
    path/media_title) within DEDUP_WINDOW_SECONDS are dropped as automation
    double-fires; returns False when deduped."""
    ensure_sessions_schema(db_path)
    at_dt = now or service_clock.legacy_now()
    at = _iso(at_dt)
    c = _connect(db_path)
    try:
        if dedup:
            row = c.execute(
                "SELECT at, payload FROM session_events "
                "WHERE session_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
                (session_id, kind),
            ).fetchone()
            if row is not None:
                try:
                    last_at = datetime.fromisoformat(row[0])
                    last_payload = json.loads(row[1] or "{}")
                except (TypeError, ValueError):
                    last_at, last_payload = None, {}
                if (last_at is not None
                        and 0 <= (at_dt - last_at).total_seconds() <= DEDUP_WINDOW_SECONDS
                        and _dedup_key(last_payload) == _dedup_key(payload)):
                    return False
        c.execute(
            "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, ?, ?)",
            (session_id, at, kind, json.dumps(payload or {}, ensure_ascii=False)),
        )
        c.execute(
            "UPDATE sessions SET last_event_at = ? WHERE session_id = ?", (at, session_id),
        )
        c.commit()
    finally:
        c.close()
    return True


def session_events(session_id: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    """All raw events for a session, oldest first (audit/test helper)."""
    ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        rows = c.execute(
            "SELECT at, kind, payload FROM session_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        c.close()
    out = []
    for at, kind, payload in rows:
        try:
            parsed = json.loads(payload or "{}")
        except ValueError:
            parsed = {}
        out.append({"at": at, "kind": kind, "payload": parsed})
    return out


# --- Outcome resolution --------------------------------------------------------
#
# Precedence when a track resolves:
#   a. explicit skip                          -> skipped
#   b. track change with ratio < SKIP_RATIO   -> skipped (inferred)
#   c. SKIP_RATIO <= ratio < COMPLETE_RATIO   -> partial
#   d. ratio >= COMPLETE_RATIO                -> played (a `complete` play event)
#   e. missing duration/position              -> partial, ratio unknown; NEVER played
#
# Deduplication: resolution is idempotent per (session_id, seq) -- once a track
# leaves 'pending', later signals are no-ops. A terminal play_events write is
# additionally guarded by a recent-identical-event window so the feedback mirror
# and session inference cannot double-write the same play.

_TERMINAL_EVENT = {"skipped": "skip", "played": "complete", "partial": "partial"}

# Resolution-reason taxonomy (WHY a track left 'pending', orthogonal to the
# outcome ladder). Step 1 is tagging only: reasons are stored everywhere a
# resolution lands but change no outcome, event kind, streak, arc, broadcast,
# or scoring behavior. Consumers filter on them in a later step.
VALID_RESOLVE_REASONS = {
    "user_skip",            # explicit skip: HA skip event / feedback mirror
    "user_skip_inferred",   # ratio-inferred at a normal track boundary
    "unplayable_suspected", # died almost immediately after starting
    "external_interruption",# boundary caused by unmatched non-library media
    "queue_replacement",    # session superseded by a new request on the player
    "session_abandoned",    # cleanup closed the session (idle or max-age)
    "pause_timeout",        # resume grace expired on a paused attachment
    "station_handoff",      # explicit Start Radio moved the station away
}

# Reasons that are SYSTEM verdicts, not user verdicts. Their 'skipped'
# outcomes are neutral: permanent history records 'partial' instead of 'skip',
# and they are excluded from every user-skip signal (skip_streak /
# skip_recovery / arc urgency / DJ observations). The session keeps the raw
# outcome + reason for diagnostics. Grows one reason per approved step.
SYSTEM_RESOLVE_REASONS = {"queue_replacement", "external_interruption",
                          "unplayable_suspected", "session_abandoned",
                          "pause_timeout", "station_handoff"}

# Rapid-failure heuristic (tag only): a track that ends within this window of
# starting, having barely progressed, is suspected unplayable (provider
# failure), not disliked. Applied only to inferred boundary resolutions --
# explicit user skips are never reclassified.
UNPLAYABLE_WINDOW_S = float(os.environ.get("PLAYBACK_UNPLAYABLE_WINDOW_S", "10"))
UNPLAYABLE_MAX_RATIO = 0.05


def _refine_reason(reason: str, outcome: str, ratio: float | None,
                   started_at: str | None, now: datetime) -> str:
    """Upgrade user_skip_inferred to unplayable_suspected only when the
    evidence is unambiguous: a skipped outcome, a known near-zero ratio, and a
    start timestamp within the rapid-failure window. Anything less stays with
    the caller's conservative reason."""
    if reason != "user_skip_inferred" or outcome != "skipped":
        return reason
    if ratio is None or ratio > UNPLAYABLE_MAX_RATIO or not started_at:
        return reason
    elapsed = service_clock.age_exact(started_at, now)
    if elapsed is None:          # unreadable, or a repeated-hour stamp
        return reason
    if 0 <= elapsed <= UNPLAYABLE_WINDOW_S:
        return "unplayable_suspected"
    return reason


def resolve_outcome(ratio: float | None, explicit_skip: bool = False) -> tuple[str, str | None]:
    """Map (position ratio, explicit flag) to (outcome, detail). Pure."""
    if explicit_skip:
        return "skipped", "explicit"
    if ratio is None:
        return "partial", "ratio_unknown"
    if ratio < SKIP_RATIO:
        return "skipped", f"ratio={ratio:.2f}<{SKIP_RATIO}"
    if ratio < COMPLETE_RATIO:
        return "partial", f"ratio={ratio:.2f}"
    return "played", f"ratio={ratio:.2f}>={COMPLETE_RATIO}"


def _stored_ratio(session: dict[str, Any]) -> float | None:
    pos, dur = session.get("last_position"), session.get("last_duration")
    if pos is None or not dur:
        return None
    try:
        return max(0.0, min(1.5, float(pos) / float(dur)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _write_terminal_play_event(path: str, artist: str, event: str, at_dt: datetime,
                               session_id: str, source: str,
                               reason: str | None = None) -> bool:
    """Write the terminal play event unless an identical terminal event for this
    path exists within the dedup window (e.g. the feedback mirror already wrote
    it). Returns True if written.

    A non-catalog item has no path and never reaches permanent play history --
    see _start_track for why that boundary exists."""
    if not path:
        return False
    import musicdb
    for row in reversed(musicdb.play_event_rows(path)):
        if row["event"] not in ("skip", "complete", "partial"):
            continue
        try:
            last = datetime.fromisoformat(row["at"])
        except (TypeError, ValueError):
            break
        if 0 <= (at_dt - last).total_seconds() <= DEDUP_WINDOW_SECONDS:
            return False
        break  # only the most recent terminal event matters
    musicdb.write_play_event(path, event, artist=artist, at=_iso(at_dt),
                             session_id=session_id, source=source,
                             detail={"reason": reason} if reason else None)
    return True


def _resolve_current(c: sqlite3.Connection, session: dict[str, Any], *,
                     ratio: float | None, explicit_skip: bool, source: str,
                     reason: str, now: datetime) -> dict[str, Any] | None:
    """Resolve the session's current track per the precedence ladder. Idempotent:
    returns None (and writes nothing) if there is no current track or it already
    left 'pending'. `reason` records WHY the resolution happened (see
    VALID_RESOLVE_REASONS); it never alters the outcome. The caller commits."""
    if reason not in VALID_RESOLVE_REASONS:
        raise ValueError(f"unknown resolve reason: {reason}")
    seq = session.get("current_seq")
    sid = session["session_id"]
    if seq is None:
        return None
    row = c.execute(
        "SELECT path, artist, outcome, started_at FROM session_tracks "
        "WHERE session_id = ? AND seq = ?", (sid, seq),
    ).fetchone()
    if row is None or row[2] != "pending":
        return None
    path, artist = row[0], row[1] or ""
    outcome, detail = resolve_outcome(ratio, explicit_skip)
    reason = _refine_reason(reason, outcome, ratio, row[3], now)
    at = _iso(now)
    c.execute(
        "UPDATE session_tracks SET outcome = ?, resolved_at = ?, resolve_reason = ? "
        "WHERE session_id = ? AND seq = ?", (outcome, at, reason, sid, seq),
    )
    c.execute(
        "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, 'track_end', ?)",
        (sid, at, json.dumps({"seq": seq, "path": path, "outcome": outcome,
                              "detail": detail, "source": source,
                              "reason": reason}, ensure_ascii=False)),
    )
    c.execute(
        "UPDATE sessions SET last_progress_at = ?, "
        "current_seq = NULL, current_path = NULL, current_media_key = '', "
        "last_position = NULL, last_duration = NULL, last_event_at = ? "
        "WHERE session_id = ?", (at, at, sid),
    )
    event = _TERMINAL_EVENT[outcome]
    if reason in SYSTEM_RESOLVE_REASONS and event == "skip":
        # A system-caused boundary (replaced queue, external media interrupting)
        # is not a user verdict on the track: permanent history records a
        # neutral 'partial' (an incomplete listen, which is literally what
        # happened); the session keeps the 'skipped' outcome + reason.
        event = "partial"
    _write_terminal_play_event(path, artist, event, now, sid,
                               source, reason=reason)
    return {"seq": seq, "path": path, "outcome": outcome, "detail": detail,
            "reason": reason}


def _recompute_observed(c: sqlite3.Connection, session_id: str) -> float | None:
    """Observed energy = mean over the last OBSERVED_ENERGY_WINDOW session tracks
    with outcome='played' AND a known energy. Played-only, binding: skipped,
    partial, pending and queued tracks never contribute. None until the first
    qualifying play (energy_fit stays neutral)."""
    rows = c.execute(
        "SELECT energy FROM session_tracks WHERE session_id = ? "
        "AND outcome = 'played' AND energy IS NOT NULL ORDER BY seq DESC LIMIT ?",
        (session_id, arcs.OBSERVED_ENERGY_WINDOW),
    ).fetchall()
    if not rows:
        return None
    return round(sum(float(r[0]) for r in rows) / len(rows), 4)


def _run_arc(c: sqlite3.Connection, session_id: str,
             now: datetime) -> dict[str, Any] | None:
    """Advance the session's arc: recompute observed energy, refresh the target,
    and apply at most one stage transition per call (tracks/minutes/predicate --
    playback_arcs decides, this applies). Every transition appends a grounded
    'arc_stage' session event. The caller commits. Returns the transition event
    payload when a transition happened, else None."""
    row = c.execute(
        "SELECT mode, arc_stage, arc_started_at, target_energy, observed_energy, "
        "context FROM sessions WHERE session_id = ?", (session_id,),
    ).fetchone()
    if row is None:
        return None
    mode, stage_key, stage_started, target, prev_observed, context_json = row
    try:
        context = json.loads(context_json or "{}")
    except (TypeError, ValueError):
        context = {}
    stage = arcs.stage_for(mode, stage_key)

    observed = _recompute_observed(c, session_id)
    if stage.get("target_energy") is None:
        # Follow-the-listener stage (Default Adaptive): target tracks observed.
        target = arcs.target_energy_for(stage, observed)
    else:
        # Banded stage: adapt when the last ADAPT_STREAK qualifying played tracks
        # all deviated > ADAPT_DEVIATION from the target on the same side --
        # follow the listener rather than force the template (may leave the band).
        recent = [float(r[0]) for r in c.execute(
            "SELECT energy FROM session_tracks WHERE session_id = ? "
            "AND outcome = 'played' AND energy IS NOT NULL ORDER BY seq DESC LIMIT ?",
            (session_id, arcs.ADAPT_STREAK)).fetchall()][::-1]
        adapted = arcs.adapt_target(target, recent)
        if adapted is not None:
            c.execute(
                "INSERT INTO session_events (session_id, at, kind, payload) "
                "VALUES (?, ?, 'arc_adapt', ?)",
                (session_id, _iso(now), json.dumps(
                    {"stage": stage["key"], "from_target": target, "to_target": adapted,
                     "recent_played_energies": recent, "observed_energy": observed},
                    ensure_ascii=False)),
            )
            target = adapted

    tracks_in_stage = c.execute(
        "SELECT COUNT(*) FROM session_tracks WHERE session_id = ? "
        "AND outcome != 'pending' AND resolved_at >= ?",
        (session_id, stage_started or ""),
    ).fetchone()[0]
    minutes_in_stage = 0.0
    if stage_started:
        bounds = service_clock.age_bounds(stage_started, now)
        if bounds is not None:
            minutes_in_stage = max(0.0, bounds[1] / 60.0)

    transition = None
    trigger = arcs.should_advance(stage, tracks_in_stage, minutes_in_stage,
                                  now=now, context=context)
    if trigger:
        nxt_key = arcs.next_stage_key(mode, stage["key"])
        if nxt_key:
            nxt = arcs.stage_for(mode, nxt_key)
            target = arcs.target_energy_for(nxt, observed)
            at = _iso(now)
            c.execute(
                "UPDATE sessions SET arc_stage = ?, arc_started_at = ? "
                "WHERE session_id = ?", (nxt_key, at, session_id),
            )
            transition = {"from": stage["key"], "to": nxt_key, "trigger": trigger,
                          "tracks_in_stage": tracks_in_stage,
                          "observed_energy": observed, "target_energy": target}
            c.execute(
                "INSERT INTO session_events (session_id, at, kind, payload) "
                "VALUES (?, ?, 'arc_stage', ?)",
                (session_id, at, json.dumps(transition, ensure_ascii=False)),
            )
    c.execute(
        "UPDATE sessions SET observed_energy = ?, target_energy = ? WHERE session_id = ?",
        (observed, target, session_id),
    )
    return transition


def _match_track(c: sqlite3.Connection, session_id: str, path: str | None,
                 media_title: str | None, media_artist: str | None,
                 media_key: str = "") -> tuple[int | None, str | None]:
    """Map an incoming track reference to a session queue row. Returns
    (seq, path): a pending queue row when matched; (None, path) for a resolved
    catalog path with no queue row; (None, None) when unresolvable.

    media_key, when the player supplied one, is tried FIRST: it is real source
    identity, so it beats title/artist guessing and it is the only thing that
    can recognise a provider item that has no catalog path at all."""
    if media_key:
        row = c.execute(
            "SELECT seq, path FROM session_tracks WHERE session_id = ? "
            "AND media_key = ? AND outcome = 'pending' ORDER BY seq LIMIT 1",
            (session_id, media_key),
        ).fetchone()
        if row:
            return row[0], (row[1] or None)
    if path:
        row = c.execute(
            "SELECT seq FROM session_tracks WHERE session_id = ? AND path = ? "
            "AND outcome = 'pending' ORDER BY seq LIMIT 1", (session_id, path),
        ).fetchone()
        return (row[0] if row else None), path
    title = (media_title or "").strip()
    artist = (media_artist or "").strip()
    if not title:
        return None, None
    # The session's own queue first: we enqueued it, so this is near-exact.
    #
    # But metadata must never merge two items the SOURCE says are different. A
    # studio take and a live take share title and artist while being distinct
    # provider items, so a row already bound to a different media_key is
    # excluded here. Rows with no key ('' -- our own queued tracks, or older
    # rows) stay eligible, which is what lets identity bind on first sight.
    key_guard = "AND (media_key = '' OR media_key = ?) " if media_key else ""
    key_args = (media_key,) if media_key else ()
    if artist:
        row = c.execute(
            "SELECT seq, path FROM session_tracks WHERE session_id = ? AND outcome = 'pending' "
            f"{key_guard}"
            "AND lower(title) = lower(?) AND lower(artist) = lower(?) ORDER BY seq LIMIT 1",
            (session_id, *key_args, title, artist),
        ).fetchone()
    else:
        row = c.execute(
            "SELECT seq, path FROM session_tracks WHERE session_id = ? AND outcome = 'pending' "
            f"{key_guard}"
            "AND lower(title) = lower(?) ORDER BY seq LIMIT 1",
            (session_id, *key_args, title),
        ).fetchone()
    if row:
        return row[0], row[1]
    # Catalog fallback (targeted, indexed); only accept an unambiguous match.
    import musicdb
    mc = musicdb.conn()
    try:
        rows = mc.execute(
            "SELECT path FROM tracks WHERE lower(artist) = lower(?) AND lower(title) = lower(?) "
            "LIMIT 2", (artist, title),
        ).fetchall() if artist else []
    finally:
        mc.close()
    if len(rows) == 1:
        return None, rows[0][0]
    return None, None


def _position_advanced(payload: dict[str, Any], session: dict[str, Any]) -> bool:
    """Did this position report carry NEW information about playback?

    Any change counts, including a seek backwards -- that is a listener
    interacting, which is liveness. Only an exactly repeated (position,
    duration) is treated as no progress. A missing position is not evidence
    either way, so it does not advance progress.
    """
    pos = payload.get("position")
    if pos is None:
        return False
    try:
        pos = float(pos)
        prev = session.get("last_position")
        prev = None if prev is None else float(prev)
    except (TypeError, ValueError):
        return False
    if prev is None:
        return True
    if pos != prev:
        return True
    dur, prev_dur = payload.get("duration"), session.get("last_duration")
    try:
        return dur is not None and prev_dur is not None \
            and float(dur) != float(prev_dur)
    except (TypeError, ValueError):
        return False


def progress_deadline_seconds(session: dict[str, Any]) -> float:
    """How long this session may go without progress before it looks wedged.

    Scaled by the duration of what is actually playing, then clamped: a
    4-minute song must move within the floor, a 3-hour recording gets
    proportionate room, and nothing gets more than the ceiling. Unknown or
    zero duration (live streams) falls back to the floor.
    """
    try:
        dur = float(session.get("last_duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    return min(max(dur * PROGRESS_DURATION_FACTOR,
                   SESSION_PROGRESS_MIN_HOURS * 3600.0),
               SESSION_PROGRESS_MAX_HOURS * 3600.0)


def resume_grace_minutes() -> float | None:
    """Customer pause grace in minutes; None = never end a pause by time.

    Read fresh on every use so a settings change applies without a restart.
    Settings must never take lifecycle down, so any read failure falls back
    to the shipped default."""
    try:
        import settings
        value = settings.load()["playback"]["resume_grace_minutes"]
        return None if value is None else float(value)
    except Exception:
        return DEFAULT_RESUME_GRACE_MINUTES


def _pause_ratio(session: dict[str, Any]) -> float | None:
    """Ratio at the frozen pause checkpoint, or None if not paused/unknown."""
    if not session.get("paused_at"):
        return None
    pos, dur = session.get("pause_position"), session.get("pause_duration")
    if pos is None or not dur:
        return None
    try:
        return max(0.0, min(1.5, float(pos) / float(dur)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _effective_ratio(session: dict[str, Any]) -> float | None:
    """Truthful resolution ratio for a system-initiated close: a PAUSED
    session resolves at its frozen pause checkpoint (the last position that
    was really heard); otherwise the stored last-position snapshot."""
    ratio = _pause_ratio(session)
    return ratio if ratio is not None else _stored_ratio(session)


def _clear_pause(c: sqlite3.Connection, session_id: str, now_dt: datetime,
                 implicit: bool = False) -> None:
    """End the paused state: same attachment, same pending track, playback
    continuing. last_progress_at restarts at the resume moment -- a resume is
    a genuine listener action, not manufactured progress, and without the
    reset a long pause would hand the progress watchdog an already-expired
    clock. The caller commits."""
    at = _iso(now_dt)
    c.execute(
        "UPDATE sessions SET paused_at = NULL, pause_position = NULL, "
        "pause_duration = NULL, last_event_at = ?, last_progress_at = ? "
        "WHERE session_id = ?", (at, at, session_id))
    if implicit:
        # A position report advanced while we still thought the player was
        # paused: playback truly resumed and the resume event was missed.
        # Recorded for forensics; identical repeats never take this path.
        c.execute(
            "INSERT INTO session_events (session_id, at, kind, payload) "
            "VALUES (?, ?, 'resumed', ?)",
            (session_id, at, json.dumps({"implicit": True})))


def session_is_live(session: dict[str, Any], now: datetime) -> bool:
    """Can this session prove it is still playing?

    Two independent signals, both required. The lease says events are
    arriving; progress says they mean something. Either alone is forgeable --
    events without progress is precisely the wedged player. A NULL
    last_event_at is never a renewal; a NULL last_progress_at falls back to
    started_at, so a session that has never progressed is judged from birth.
    """
    last_event = session.get("last_event_at")
    if not last_event:
        return False
    # DST-safe: a session is judged dead only when its stamps are
    # CERTAINLY older than the lease / progress deadline.
    if last_event < service_clock.cutoff_destructive(
            now, timedelta(minutes=SESSION_LEASE_MINUTES)):
        return False
    last_progress = session.get("last_progress_at") or session.get("started_at")
    if not last_progress:
        return False
    cutoff = service_clock.cutoff_destructive(
        now, timedelta(seconds=progress_deadline_seconds(session)))
    return last_progress >= cutoff


def _token_shaped(token: str) -> bool:
    return (len(token or "") == 32
            and all(ch in "0123456789abcdef" for ch in token))


#: The three words the playback path can say about a programme item that
#: just became Music Assistant's current item, decided by the SAME answer
#: the audio route gives for its token -- what the player was served.
PROGRAMME_LIVE = "live"          # real audio: a break or a bulletin on air
PROGRAMME_RETIRED = "retired"    # the retired-silence tombstone: a spent replay
PROGRAMME_UNKNOWN = "unknown"    # 404 (never ours, or forgotten) or no answer


def _programme_verdict(kind: str, token: str, db_path=None) -> tuple[str, str]:
    """(verdict, evidence) for the programme item that just became MA's
    current item: PROGRAMME_LIVE, PROGRAMME_RETIRED or PROGRAMME_UNKNOWN.

    Decided by the serving gate itself (`dj_engine.audio_available`; for a
    bulletin, `news_bulletins.audio_available` reached through
    `news_schedule.bulletin_audio_available`) -- the one place that knows whether
    this token answers real audio or the retired-silence tombstone -- so
    the airing record can never disagree with what the player was served.
    The retirement ledger alone is not enough: a break
    keeps a TERMINAL row for a day before the reaper writes its retirement
    fact, an expired bulletin's row is retired only at cleanup, and a
    bulletin past the age gate is silence by the clock. The gate answers
    the tombstone for all of them, and only the gate does.

    Fails to UNKNOWN, never to LIVE: a gate that cannot answer, or answers
    404 (which is what the player got, so nothing played), has not
    confirmed a broadcast, and an airing is recorded only on a confirmed
    one. A LIVE row answers live whatever the ledger's state, so nothing
    the route would serve is ever suppressed here."""
    if not _token_shaped(token):
        return PROGRAMME_UNKNOWN, "malformed"
    try:
        if kind == "break":
            import dj_engine
            status, path = dj_engine.audio_available(token, db_path)
        else:
            # through the scheduler: the session store must not depend on
            # the bulletin store (test_news_actions, DarknessTests)
            import news_schedule
            status, path = news_schedule.bulletin_audio_available(token, db_path)
    except Exception:  # noqa: BLE001 - never costs event ingestion
        return PROGRAMME_UNKNOWN, "gate_unreadable"
    if status == 200:
        import program_audio
        if program_audio.is_tombstone(path):
            return PROGRAMME_RETIRED, "tombstone"
        return PROGRAMME_LIVE, "served"
    if status == 410:
        return PROGRAMME_RETIRED, "gone"
    return PROGRAMME_UNKNOWN, "not_served"


def _prior_airing(c: sqlite3.Connection, token: str) -> bool | None:
    """Did this token go on air for real at some earlier moment -- a
    dj_break or news_bulletin event, which is written only when a LIVE
    token became the current item? True / False, or None when the token
    is not even token-shaped (nothing can be said). Distinguishes a spent
    item that genuinely aired before its retirement from one that never
    aired at all; both now replay as a tombstone, and neither replay is an
    airing."""
    if not _token_shaped(token):
        return None
    try:
        row = c.execute(
            "SELECT 1 FROM session_events WHERE kind IN "
            "('dj_break', 'news_bulletin') AND payload LIKE ? LIMIT 1",
            (f'%"token": "{token}"%',)).fetchone()
    except sqlite3.Error:
        return None
    return row is not None


def handle_event(session_id: str, kind: str, payload: dict[str, Any] | None = None,
                 now: datetime | None = None,
                 db_path: Path | None = None) -> dict[str, Any]:
    """Ingest one HA/MA player event. Kinds: track_start, position, skip, idle,
    stop, paused, resumed. Returns {ok, deduped?, resolved?, started?,
    paused?, resumed?}."""
    ensure_sessions_schema(db_path)
    payload = payload or {}
    now_dt = now or service_clock.legacy_now()
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "session not found"}
        session = _session_dict(row)
        if session["status"] != "active":
            return {"ok": False, "error": "session not active"}
    finally:
        c.close()

    # Raw event log first (with the double-fire window). A deduped raw event is
    # dropped entirely -- it is the same physical occurrence.
    #
    # A position heartbeat is NOT logged as an event row. It is an
    # observation, not a transition: the one fact it carries (where the
    # player is, and that it is still reporting) lands on the session row
    # below, which is the only place liveness ever reads it. Logging it too
    # meant one appended row and a second commit every thirty seconds of
    # every listening hour, for a row nothing read back (a storage
    # endurance cost). Transitions -- track_start, paused, resumed,
    # stop, idle, skip -- are still logged exactly as before.
    if kind != "position" and not append_event(session_id, kind, payload,
                                               now=now_dt, db_path=db_path):
        return {"ok": True, "deduped": True}

    c = _connect(db_path)
    try:
        result: dict[str, Any] = {"ok": True}
        if kind == "position":
            # The wedge detector. A position report that repeats the previous
            # one renews the EVENT heartbeat but is not evidence that playback
            # advanced -- a frozen player keeps sending these on the 30s timer.
            at = _iso(now_dt)
            if _position_advanced(payload, session):
                if session.get("paused_at"):
                    # Genuine advancement while flagged paused: playback really
                    # resumed and the resume event was missed. Advancement is
                    # evidence; identical repeats (the else branch) can never
                    # manufacture a resume out of a heartbeat.
                    _clear_pause(c, session_id, now_dt, implicit=True)
                    result["resumed"] = {"implicit": True}
                c.execute(
                    "UPDATE sessions SET last_position = ?, last_duration = ?, "
                    "last_event_at = ?, last_progress_at = ? WHERE session_id = ?",
                    (payload.get("position"), payload.get("duration"), at, at,
                     session_id),
                )
            else:
                c.execute(
                    "UPDATE sessions SET last_position = ?, last_duration = ?, "
                    "last_event_at = ? WHERE session_id = ?",
                    (payload.get("position"), payload.get("duration"), at,
                     session_id),
                )
        elif kind == "paused":
            # Playback intentionally suspended: the attachment stays logically
            # alive, the current track stays pending, and the position checkpoint
            # (payload position verbatim, else the stored snapshot) is frozen as
            # the truthful pause position. Nothing resolves here.
            at = _iso(now_dt)
            if session.get("paused_at"):
                # Already paused: heartbeat only. The ORIGINAL checkpoint and
                # pause time stand -- repeated paused reports must not slide
                # the grace window or the position.
                c.execute("UPDATE sessions SET last_event_at = ? "
                          "WHERE session_id = ?", (at, session_id))
            else:
                pos = payload.get("position")
                dur = payload.get("duration")
                if pos is None:
                    pos = session.get("last_position")
                if dur is None:
                    dur = session.get("last_duration")
                c.execute(
                    "UPDATE sessions SET paused_at = ?, pause_position = ?, "
                    "pause_duration = ?, last_event_at = ? WHERE session_id = ?",
                    (at, pos, dur, at, session_id))
            result["paused"] = True
        elif kind == "resumed":
            # Same queue resuming within grace: same session, same pending
            # track, no artificial boundary. Recognised even when media_title
            # never changed, because this is an explicit transition event.
            if session.get("paused_at"):
                _clear_pause(c, session_id, now_dt)
                result["resumed"] = {"implicit": False}
            else:
                # Not paused: a stray resume is a heartbeat, nothing more.
                c.execute("UPDATE sessions SET last_event_at = ? "
                          "WHERE session_id = ?", (_iso(now_dt), session_id))
                result["resumed"] = None
        elif kind == "skip":
            resolved = _resolve_current(c, session, ratio=None, explicit_skip=True,
                                        source="explicit", reason="user_skip",
                                        now=now_dt)
            result["resolved"] = resolved
        elif kind == "idle":
            if session.get("paused_at"):
                # MA represents a long user pause as paused -> idle. That
                # transition is a REPRESENTATION change, not new evidence the
                # attachment died: the pause grace governs. Heartbeat only.
                c.execute("UPDATE sessions SET last_event_at = ? "
                          "WHERE session_id = ?", (_iso(now_dt), session_id))
                result["ignored"] = "idle_while_paused"
            else:
                ratio = _ratio_from(payload, session)
                resolved = _resolve_current(c, session, ratio=ratio,
                                            explicit_skip=False, source="inferred",
                                            reason="user_skip_inferred", now=now_dt)
                result["resolved"] = resolved
        elif kind == "stop":
            # Explicit stop/off closes by the truthful position ladder NOW --
            # never waiting out a pause grace. For a paused session the frozen
            # pause checkpoint IS the truth; later payloads may carry stale or
            # zeroed positions from the dying player.
            if session.get("paused_at"):
                ratio = _pause_ratio(session)
                resolved = _resolve_current(c, session, ratio=ratio,
                                            explicit_skip=False, source="inferred",
                                            reason="user_skip_inferred", now=now_dt)
                c.execute(
                    "UPDATE sessions SET paused_at = NULL, pause_position = NULL, "
                    "pause_duration = NULL WHERE session_id = ?", (session_id,))
            else:
                ratio = _ratio_from(payload, session)
                resolved = _resolve_current(c, session, ratio=ratio,
                                            explicit_skip=False, source="inferred",
                                            reason="user_skip_inferred", now=now_dt)
            result["resolved"] = resolved
        elif kind == "track_start":
            prog_kind, prog_token = media_ref.programming_ref(
                payload.get("media_content_id"))
            if prog_kind:
                # PROGRAMMING audio became the current MA item -- the DJ's own
                # break, or a news bulletin this process cached and served.
                # Either way it is not music: it never becomes a session
                # track, never enters play_events, never disturbs
                # current_media_key beyond the ordinary boundary bookkeeping,
                # and never supersedes the session. The finished song resolves
                # through its normal natural-boundary evidence; repeats are
                # no-ops (the current track has already left 'pending').
                ratio = _ratio_from(payload, session, previous=True)
                resolved = _resolve_current(c, session, ratio=ratio,
                                            explicit_skip=False,
                                            source="inferred",
                                            reason="user_skip_inferred",
                                            now=now_dt)
                result["resolved"] = resolved
                verdict, evidence = _programme_verdict(prog_kind, prog_token,
                                                       db_path)
                if verdict != PROGRAMME_LIVE:
                    # THE TOMBSTONE RULE. A retired token is served as a few
                    # seconds of silence (program_tokens, program_audio), so
                    # what just became MA's current item is NOT a break or a
                    # bulletin going on air: it is a spent queue item hygiene
                    # had not yet removed, replaying as a tombstone. Writing
                    # the ordinary airing event here is how a silent replay
                    # became a line in the listener's Break History, and the
                    # DJ/News hooks below would re-arm a bulletin's airing
                    # state or stamp an air start for something nobody heard.
                    # So: the music boundary above still resolves (the song
                    # really did end), a DISTINCT event records the replay
                    # with whether the token ever aired for real, and no
                    # programme hook runs. Real history is never rewritten:
                    # an earlier genuine airing keeps its own row.
                    #
                    # UNKNOWN is the same decision for a different reason:
                    # the gate answered 404 (so did the player's fetch, and
                    # nothing played) or could not answer at all. Neither
                    # confirms a broadcast, so neither is recorded as one;
                    # the `program_unverified` event keeps the fact and the
                    # reason instead of inventing an airing or losing the
                    # moment. Only a confirmed LIVE answer takes the
                    # ordinary path below.
                    prior = _prior_airing(c, prog_token)
                    note = {"kind": prog_kind, "token": prog_token,
                            "prior_airing": prior}
                    if verdict == PROGRAMME_RETIRED:
                        event_kind, key = "program_tombstone", "tombstone"
                    else:
                        event_kind, key = "program_unverified", "unverified"
                        note["evidence"] = evidence
                    result[key] = dict(note)
                    c.execute(
                        "INSERT INTO session_events (session_id, at, kind, payload) "
                        "VALUES (?, ?, ?, ?)",
                        (session_id, _iso(now_dt), event_kind,
                         json.dumps(note)))
                    c.commit()
                    c.close()
                    return result
                if prog_kind == "break":
                    result["break_airing"] = prog_token
                    event_kind = "dj_break"
                else:
                    result["programming_airing"] = {"kind": prog_kind,
                                                    "token": prog_token}
                    event_kind = "news_bulletin"
                c.execute(
                    "INSERT INTO session_events (session_id, at, kind, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, _iso(now_dt), event_kind,
                     json.dumps({"token": prog_token})))
                c.commit()
                c.close()
                # DJ bookkeeping strictly AFTER commit (its own connection
                # must never contend with this one) and strictly fenced. Only
                # a BREAK is DJ bookkeeping: a bulletin is not one of the DJ's
                # breaks and must not close one out or move the cadence anchor.
                if prog_kind == "break":
                    try:
                        import dj_engine
                        dj_engine.on_break_track_start(prog_token, now_dt,
                                                       db_path=db_path)
                    except Exception:
                        pass
                else:
                    # The publisher bulletin is on air. This is the moment
                    # the news event records that this token and this target
                    # hour are consumed -- written NOW, not five minutes
                    # later, so a restart mid-bulletin can never replay it.
                    try:
                        import news_schedule
                        news_schedule.on_bulletin_start(prog_token, now_dt,
                                                        db_path=db_path)
                    except Exception:
                        pass
                return result
            incoming_key = media_ref.media_key(payload.get("media_content_id"))
            seq, path = _match_track(c, session_id, payload.get("path"),
                                     payload.get("media_title"),
                                     payload.get("media_artist"),
                                     media_key=incoming_key)
            same_track = (
                (incoming_key and incoming_key == session.get("current_media_key"))
                or (path is not None and path == session.get("current_path")))
            if same_track:
                # Same track re-announced: not a boundary. media_key is
                # checked first so a provider item with no catalog path still
                # deduplicates -- several group members announcing one
                # streaming-provider track must not become several boundaries.
                #
                # Deliberately NOT a resume for a paused session: a stale or
                # duplicated start event carries identical metadata to a real
                # one, so identity alone is not evidence playback resumed.
                # Resumes require the explicit 'resumed' transition or an
                # ADVANCING position report -- both duplication-proof.
                result["started"] = None
            else:
                # A boundary we cannot attribute to the listener: something
                # EXTERNAL displaced the current track. Note this is now judged
                # on IDENTITY, not on catalog membership -- a provider item is
                # identifiable media, not an interruption.
                identifiable = (path is not None) or bool(incoming_key)
                if session.get("paused_at"):
                    # A different track starting on a paused session: the
                    # interrupted track resolves at its frozen pause checkpoint
                    # (the payload's previous_* may be stale from the pause),
                    # and the pause itself is over.
                    ratio = _pause_ratio(session)
                    _clear_pause(c, session_id, now_dt)
                else:
                    ratio = _ratio_from(payload, session, previous=True)
                resolved = _resolve_current(
                    c, session, ratio=ratio, explicit_skip=False,
                    source="inferred",
                    reason=("user_skip_inferred" if identifiable
                            else "external_interruption"),
                    now=now_dt)
                result["resolved"] = resolved
                result["started"] = _start_track(c, session_id, seq, path, payload, now_dt)
        else:
            return {"ok": False, "error": f"unknown event kind: {kind}"}
        if kind != "position":
            # Arc bookkeeping after any potential resolution/boundary: observed
            # energy (played-only window), target refresh, stage transitions.
            transition = _run_arc(c, session_id, now_dt)
            if transition:
                result["arc_transition"] = transition
        c.commit()
    finally:
        c.close()
    if kind == "track_start" and result.get("started") is not None:
        # Queue-native DJ hook, strictly after commit and strictly fenced:
        # completes an aired break, and (only when dj.enabled permits this
        # queue) runs an airtime-aware scheduling pass. No failure in here
        # may ever reach event ingestion.
        try:
            import dj_engine
            dj_engine.on_music_track_start(
                session, str(payload.get("queue_id") or session.get("queue_id")
                             or ""), now_dt,
                current={"artist": payload.get("media_artist") or "",
                         "title": payload.get("media_title") or ""},
                db_path=db_path)
        except Exception:
            pass
    elif kind == "position":
        # Position heartbeats are the DJ's render-window clock: the tick
        # hook is how spoken text gets generated ~2 minutes before the
        # target boundary instead of at a stale earlier boundary. Strictly
        # after commit and strictly fenced, exactly like the boundary hook;
        # the hook itself is inert unless dj.enabled permits this queue.
        try:
            import dj_engine
            dj_engine.on_playback_tick(
                session, str(payload.get("queue_id") or session.get("queue_id")
                             or ""), now_dt, db_path=db_path)
        except Exception:
            pass
    return result


def _ratio_from(payload: dict[str, Any], session: dict[str, Any],
                previous: bool = False) -> float | None:
    """Ratio for resolving the *current* track: prefer the payload's explicit
    previous_position/previous_duration (HA from_state), else position/duration
    on non-boundary events, else the session's stored last position snapshot."""
    if previous:
        pos, dur = payload.get("previous_position"), payload.get("previous_duration")
    else:
        pos, dur = payload.get("position"), payload.get("duration")
    if pos is not None and dur:
        try:
            return max(0.0, min(1.5, float(pos) / float(dur)))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return _stored_ratio(session)


def _start_track(c: sqlite3.Connection, session_id: str, seq: int | None,
                 path: str | None, payload: dict[str, Any],
                 now_dt: datetime) -> dict[str, Any] | None:
    """Mark a track as now playing: a matched pending queue row, a resolved
    catalog path (ad-hoc row, source 'external'), or an unmatched title (ad-hoc
    row with an empty-path placeholder is NOT created -- unresolvable starts are
    logged and skipped)."""
    at = _iso(now_dt)
    ref = media_ref.describe(payload, path)
    if seq is None and path is None and not ref["media_key"]:
        # Genuinely unidentifiable: no catalog path AND no source identity.
        # There is nothing honest to record, so it stays an event, as before.
        c.execute(
            "INSERT INTO session_events (session_id, at, kind, payload) VALUES (?, ?, 'unmatched', ?)",
            (session_id, at, json.dumps({"media_title": payload.get("media_title"),
                                         "media_artist": payload.get("media_artist")},
                                        ensure_ascii=False)),
        )
        return None
    artist = payload.get("media_artist") or ""
    if seq is None:
        row = c.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM session_tracks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row[0]) + 1
        # path is '' for a provider item: the column means "which library row",
        # and there is none. Identity is carried by media_key instead.
        c.execute(
            "INSERT INTO session_tracks (session_id, seq, path, artist, album, title, "
            "queued_at, started_at, outcome, source, media_key, media_uri, "
            "media_namespace, match_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'external', ?, ?, ?, ?)",
            (session_id, seq, path or "", artist, ref["album"],
             payload.get("media_title") or "", at, at,
             ref["media_key"], ref["media_uri"], ref["media_namespace"],
             ref["match_status"]),
        )
    else:
        c.execute(
            "UPDATE session_tracks SET started_at = ? WHERE session_id = ? AND seq = ?",
            (at, session_id, seq),
        )
        if ref["media_key"]:
            # A queued row we have just recognised: bind identity to it once,
            # without disturbing a row that already carries one.
            c.execute(
                "UPDATE session_tracks SET media_key = ?, media_uri = ?, "
                "media_namespace = ?, match_status = ? "
                "WHERE session_id = ? AND seq = ? AND media_key = ''",
                (ref["media_key"], ref["media_uri"], ref["media_namespace"],
                 ref["match_status"], session_id, seq),
            )
        row = c.execute(
            "SELECT artist FROM session_tracks WHERE session_id = ? AND seq = ?",
            (session_id, seq),
        ).fetchone()
        artist = (row[0] if row else "") or artist
    c.execute(
        "UPDATE sessions SET current_seq = ?, current_path = ?, "
        "current_media_key = ?, "
        "last_position = ?, last_duration = ?, last_event_at = ?, "
        "last_progress_at = ?, queue_depth = "
        "(SELECT COUNT(*) FROM session_tracks WHERE session_id = ? AND outcome = 'pending' "
        "AND started_at IS NULL) WHERE session_id = ?",
        (seq, path, ref["media_key"], payload.get("position"),
         payload.get("duration"), at, at, session_id, session_id),
    )
    # Permanent play history stays catalog-keyed: play_events.path is a local
    # library path and every learning consumer (familiarity, forgotten lane,
    # feedback, scoring) joins on it. A provider item has no such path, and
    # inventing one is exactly what this work exists to avoid. External items
    # therefore live fully in session history and do NOT enter play_events.
    if path:
        import musicdb
        musicdb.write_play_event(path, "start", artist=artist, at=at,
                                 session_id=session_id, source="inferred")
    return {"seq": seq, "path": path}


def mirror_feedback(action: str, path: str | None = None, artist: str | None = None,
                    title: str | None = None, session_id: str | None = None,
                    now: datetime | None = None,
                    db_path: Path | None = None) -> list[dict[str, Any]]:
    """Mirror a /feedback call into active sessions whose *current* track matches
    (or into one explicit session). Explicit skip resolves the current track with
    precedence (a); thumbs are recorded as session events only. Best-effort: the
    caller must treat failures as non-fatal."""
    ensure_sessions_schema(db_path)
    now_dt = now or service_clock.legacy_now()
    c = _connect(db_path)
    try:
        if session_id:
            rows = c.execute(
                f"SELECT {', '.join(_SESSION_COLS)} FROM sessions "
                "WHERE session_id = ? AND status = 'active'", (session_id,),
            ).fetchall()
        else:
            rows = c.execute(
                f"SELECT {', '.join(_SESSION_COLS)} FROM sessions WHERE status = 'active'",
            ).fetchall()
        sessions = [_session_dict(r) for r in rows]
    finally:
        c.close()

    results = []
    for session in sessions:
        sid = session["session_id"]
        matched = False
        if session.get("current_seq") is not None:
            if path:
                matched = session.get("current_path") == path
            elif artist and title:
                c = _connect(db_path)
                try:
                    row = c.execute(
                        "SELECT 1 FROM session_tracks WHERE session_id = ? AND seq = ? "
                        "AND lower(artist) = lower(?) AND lower(title) = lower(?)",
                        (sid, session["current_seq"], artist, title),
                    ).fetchone()
                finally:
                    c.close()
                matched = row is not None
        if not matched and not session_id:
            continue
        append_event(sid, action, {"path": path, "artist": artist, "title": title,
                                   "matched_current": matched},
                     now=now_dt, db_path=db_path)
        resolved = None
        if action == "skip" and matched:
            c = _connect(db_path)
            try:
                resolved = _resolve_current(c, session, ratio=None, explicit_skip=True,
                                            source="explicit", reason="user_skip",
                                            now=now_dt)
                c.commit()
            finally:
                c.close()
        results.append({"session_id": sid, "matched_current": matched,
                        "resolved": resolved})
    return results


# --- Track features: derived, disposable, estimated-and-honest --------
#
# pi_track_features materializes per-track scoring inputs into collection.db
# (knowledge-layer placement). Energy/mood are LEXICON PROXIES with provenance
# 'estimated' -- no audio analysis exists and none is added; observed skips are
# the corrective. meta_confidence shrinks estimated components toward neutral.

import re as _re

FEATURES_SCHEMA_VERSION = 1

# Controlled genre vocabulary: raw tag (lowered) -> normalized bucket.
_GENRE_NORM = {
    "metal": "metal", "heavy metal": "metal", "thrash": "metal", "nu metal": "metal",
    "punk": "punk", "pop punk": "punk", "hardcore": "punk",
    "hard rock": "hard_rock", "grunge": "grunge", "alternative": "alternative",
    "alt rock": "alternative", "alternative rock": "alternative", "indie": "alternative",
    "rock": "rock", "classic rock": "rock", "southern rock": "rock",
    "pop": "pop", "dance": "dance", "electronic": "dance", "edm": "dance",
    "house": "dance", "techno": "dance",
    "hip hop": "hip_hop", "hip-hop": "hip_hop", "rap": "hip_hop",
    "country": "country", "folk": "folk", "singer-songwriter": "folk",
    "acoustic": "acoustic", "blues": "blues", "jazz": "jazz",
    "classical": "classical", "soundtrack": "soundtrack", "ambient": "ambient",
    "r&b": "rnb", "soul": "rnb", "funk": "rnb", "reggae": "reggae", "ska": "reggae",
}

# Energy proxy base per normalized genre (documented lexicon; 0..1, estimated).
_GENRE_ENERGY = {
    "metal": 0.85, "punk": 0.85, "dance": 0.80, "hard_rock": 0.75, "grunge": 0.70,
    "hip_hop": 0.70, "rock": 0.65, "alternative": 0.60, "pop": 0.60, "reggae": 0.55,
    "rnb": 0.50, "country": 0.50, "blues": 0.45, "soundtrack": 0.45, "folk": 0.35,
    "jazz": 0.35, "acoustic": 0.30, "classical": 0.20, "ambient": 0.15,
}

_CALM_TITLE_RE = _re.compile(r"\b(?:ballad|lullaby|acoustic|unplugged|interlude)\b")
_HYPE_TITLE_RE = _re.compile(r"\b(?:party|dance|anthem|rocker)\b")
_LIVE_RE = _re.compile(r"\b(?:live|unplugged)\b")
_DEMO_RE = _re.compile(r"\b(?:demo|rehearsal|rough mix|early mix)\b")
_INSTRUMENTAL_RE = _re.compile(r"\binstrumental\b")


def _norm_text(text: str) -> str:
    return _re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _normalize_genre(raw: str) -> str | None:
    text = _norm_text(raw)
    if not text:
        return None
    if text in _GENRE_NORM:
        return _GENRE_NORM[text]
    for token, norm in _GENRE_NORM.items():
        if _re.search(rf"\b{_re.escape(token)}\b", text):
            return norm
    return "other"


def _decade(year_text: Any) -> int | None:
    digits = _re.match(r"\s*(\d{4})", str(year_text or ""))
    if not digits:
        return None
    year = int(digits.group(1))
    if not 1900 <= year <= 2100:
        return None
    return (year // 10) * 10


def compute_track_features(t: dict[str, Any]) -> dict[str, Any]:
    """Pure: one catalog track row -> its feature row. Energy/mood are estimated
    lexicon proxies; flags are word-boundary matches (release_filters-proven)."""
    title_album = _norm_text(f"{t.get('title') or ''} {t.get('album') or ''}")
    genre_raw = (t.get("genre") or "").strip()
    genre_norm = _normalize_genre(genre_raw)
    decade = _decade(t.get("year"))
    is_live = 1 if _LIVE_RE.search(title_album) else 0
    is_demo = 1 if _DEMO_RE.search(title_album) else 0
    is_instr = 1 if (_INSTRUMENTAL_RE.search(title_album)
                     or _INSTRUMENTAL_RE.search(_norm_text(genre_raw))) else 0

    energy = _GENRE_ENERGY.get(genre_norm or "", 0.5)
    if _CALM_TITLE_RE.search(title_album):
        energy -= 0.15
    if _HYPE_TITLE_RE.search(title_album):
        energy += 0.10
    if is_live:
        energy += 0.05
    if is_instr:
        energy -= 0.05
    energy = max(0.0, min(1.0, round(energy, 3)))

    if energy >= 0.70:
        mood = "energetic"
    elif energy <= 0.35:
        mood = "mellow"
    elif genre_norm in ("pop", "dance", "reggae"):
        mood = "upbeat"
    else:
        mood = "neutral"

    known_genre = genre_norm is not None and genre_norm != "other"
    meta_confidence = round(
        0.4 * (1 if known_genre else 0)
        + 0.3 * (1 if decade is not None else 0)
        + 0.2 * (1 if (t.get("mb_recording_id") or t.get("mb_release_group_id")) else 0)
        + 0.1 * (1 if t.get("duration") else 0), 3)

    return {
        "path": t.get("path"),
        "artist": t.get("artist") or "",
        "album": t.get("album") or "",
        "title": t.get("title") or "",
        "genre_norm": genre_norm,
        "genre_raw": genre_raw,
        "decade": decade,
        "duration": t.get("duration"),
        "is_live": is_live,
        "is_demo": is_demo,
        "is_instrumental": is_instr,
        "energy": energy,
        "mood": mood,
        "meta_confidence": meta_confidence,
        "provenance": json.dumps({
            "genre_norm": "grounded_local" if known_genre else "neutral",
            "decade": "grounded_local" if decade is not None else "neutral",
            "flags": "grounded_local",
            "energy": "estimated",
            "mood": "estimated",
        }),
    }


_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS pi_track_features (
    path TEXT PRIMARY KEY,
    artist TEXT DEFAULT '',
    album TEXT DEFAULT '',
    title TEXT DEFAULT '',
    genre_norm TEXT,
    genre_raw TEXT DEFAULT '',
    decade INTEGER,
    duration REAL,
    is_live INTEGER DEFAULT 0,
    is_demo INTEGER DEFAULT 0,
    is_instrumental INTEGER DEFAULT 0,
    energy REAL,
    mood TEXT,
    meta_confidence REAL,
    provenance TEXT DEFAULT '{}',
    built_at TEXT
)
"""


def build_features(collection_db: Path | None = None, now: datetime | None = None,
                   tracks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Materialize pi_track_features (derived, disposable, regenerable). Reads
    the catalog via musicdb; full rebuild each run (single pass, cheap)."""
    if tracks is None:
        import musicdb
        c = musicdb.conn()
        try:
            cur = c.execute(
                "SELECT path, artist, album, title, genre, year, duration, "
                "mb_recording_id, mb_release_group_id FROM tracks")
            cols = [d[0] for d in cur.description]
            tracks = [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            c.close()
    at = _iso(now)
    rows = [compute_track_features(t) for t in tracks if t.get("path")]
    db = Path(collection_db) if collection_db is not None else COLLECTION_DB
    c = sqlite3.connect(db)
    try:
        c.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        c.execute(_FEATURES_DDL)
        c.execute("DELETE FROM pi_track_features")
        c.executemany(
            "INSERT OR REPLACE INTO pi_track_features "
            "(path, artist, album, title, genre_norm, genre_raw, decade, duration, "
            "is_live, is_demo, is_instrumental, energy, mood, meta_confidence, "
            "provenance, built_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r["path"], r["artist"], r["album"], r["title"], r["genre_norm"],
              r["genre_raw"], r["decade"], r["duration"], r["is_live"], r["is_demo"],
              r["is_instrumental"], r["energy"], r["mood"], r["meta_confidence"],
              r["provenance"], at) for r in rows],
        )
        c.commit()
    finally:
        c.close()
    known = sum(1 for r in rows if r["genre_norm"] not in (None, "other"))
    dec = sum(1 for r in rows if r["decade"] is not None)
    conf = [r["meta_confidence"] for r in rows] or [0]
    return {
        "tracks": len(rows),
        "genre_known_pct": round(100.0 * known / len(rows), 1) if rows else 0.0,
        "decade_known_pct": round(100.0 * dec / len(rows), 1) if rows else 0.0,
        "mean_confidence": round(sum(conf) / len(conf), 3),
        "built_at": at,
    }


# Request parsing kept from the retired retrieval section: session_scoring_state
# still derives request tokens and the decade window for live state readers.
_DECADE_WINDOWS = {
    ("60s", "1960s", "sixties"): (1960, 1969),
    ("70s", "1970s", "seventies"): (1970, 1979),
    ("80s", "1980s", "eighties"): (1980, 1989),
    ("90s", "1990s", "nineties"): (1988, 2000),
    ("2000s", "aughts", "noughties"): (2000, 2009),
    ("2010s",): (2010, 2019),
}


def _request_tokens(request: str) -> set[str]:
    return set(_re.findall(r"[a-z0-9]+", (request or "").lower()))


def _request_decade_window(tokens: set[str]) -> tuple[int, int] | None:
    for keys, window in _DECADE_WINDOWS.items():
        if tokens & set(keys):
            return window
    return None




def session_scoring_state(session_id: str, db_path: Path | None = None,
                          collection_db: Path | None = None,
                          candidate_paths: list[str] | None = None,
                          candidate_artists: list[str] | None = None) -> dict[str, Any]:
    """Bounded read of a session's live scoring state (seed, request, energy,
    outcome history, skip_recovery_armed). One sessions.db pass + two targeted
    persistent-store queries + one knowledge read. The retired track scorer was
    its original consumer; it remains the sanctioned observation seam for
    skip-recovery and session-history state (tests and future consumers)."""
    import musicdb
    c = _connect(db_path)
    try:
        row = c.execute(
            "SELECT seed, request, target_energy, observed_energy FROM sessions "
            "WHERE session_id = ?", (session_id,)).fetchone()
        seed, request, target_energy, observed = row if row else (0, "", None, None)
        resolved = c.execute(
            "SELECT artist, album, genre, year, outcome, resolve_reason "
            "FROM session_tracks "
            "WHERE session_id = ? AND outcome != 'pending' ORDER BY seq",
            (session_id,),
        ).fetchall()
        queued = c.execute(
            "SELECT artist, album FROM session_tracks WHERE session_id = ? "
            "AND outcome = 'pending' ORDER BY seq", (session_id,),
        ).fetchall()
        n_rows = c.execute(
            "SELECT COUNT(*) FROM session_tracks WHERE session_id = ?",
            (session_id,)).fetchone()[0]
    finally:
        c.close()

    plays = [{"artist": r[0] or "", "album": r[1] or "", "genre": r[2] or "",
              "decade": _decade(r[3]), "outcome": r[4],
              "resolve_reason": r[5]} for r in resolved]
    played = [p for p in plays if p["outcome"] in ("played", "partial")]
    recent_artists = [p["artist"].lower() for p in plays] + \
                     [(q[0] or "").lower() for q in queued]
    recent_albums = [f"{p['artist']}|{p['album']}".lower() for p in plays] + \
                    [f"{(q[0] or '')}|{(q[1] or '')}".lower() for q in queued]
    outcomes = [p["outcome"] for p in plays]
    skip_streak = 0
    for p in reversed(plays):
        # A system-reasoned 'skipped' is not a user skip: it neither extends
        # the streak nor passes through it (treated like any non-skip outcome,
        # so it never arms skip_recovery or the arc's skip urgency).
        if p["outcome"] == "skipped" and \
                p["resolve_reason"] not in SYSTEM_RESOLVE_REASONS:
            skip_streak += 1
        else:
            break

    # Affinity for only the candidate artists (bounded).
    affinity: dict[str, float] = {}
    db = Path(collection_db) if collection_db is not None else COLLECTION_DB
    if candidate_artists and db.exists():
        conn_a = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            try:
                ph = ",".join("?" for _ in candidate_artists)
                for name, score in conn_a.execute(
                        f"SELECT name, affinity_score FROM ci_affinity "
                        f"WHERE lower(name) IN ({ph})",
                        [a.lower() for a in candidate_artists]).fetchall():
                    affinity[name.lower()] = float(score or 0.5)
            except sqlite3.OperationalError:
                pass
        finally:
            conn_a.close()

    # Complete-play counts + last terminal timestamps for candidates (bounded).
    completes: dict[str, int] = {}
    last_play: dict[str, str] = {}
    if candidate_paths:
        mc = musicdb.conn()
        try:
            ph = ",".join("?" for _ in candidate_paths)
            for path, n in mc.execute(
                    f"SELECT path, COUNT(*) FROM play_events "
                    f"WHERE event = 'complete' AND path IN ({ph}) GROUP BY path",
                    candidate_paths).fetchall():
                completes[path] = n
            for path, at in mc.execute(
                    f"SELECT path, MAX(at) FROM play_events "
                    f"WHERE event IN ('start','complete','skip','partial') "
                    f"AND path IN ({ph}) GROUP BY path", candidate_paths).fetchall():
                last_play[path] = at
        finally:
            mc.close()

    genre_counts: dict[str, int] = {}
    decade_counts: dict[str, int] = {}
    artist_play_counts: dict[str, int] = {}
    for p in played:
        g = _normalize_genre(p["genre"]) or "unknown"
        genre_counts[g] = genre_counts.get(g, 0) + 1
        d = str(p["decade"] or "unknown")
        decade_counts[d] = decade_counts.get(d, 0) + 1
        artist_play_counts[p["artist"].lower()] = \
            artist_play_counts.get(p["artist"].lower(), 0) + 1
    # Intended-queue artist counts: projected diversity considers tracks
    # already queued, not just played -- a pre-built queue can't hide an artist.
    artist_queued_counts: dict[str, int] = {}
    for q in queued:
        a = (q[0] or "").lower()
        artist_queued_counts[a] = artist_queued_counts.get(a, 0) + 1

    tokens = _request_tokens(request or "")
    return {
        "seed": int(seed or 0),
        "request": request or "",
        "tokens": tokens,
        "decade_window": _request_decade_window(tokens),
        "target_energy": target_energy if target_energy is not None else observed,
        "recent_artists": recent_artists,
        "recent_albums": recent_albums,
        "genre_counts": genre_counts,
        "decade_counts": decade_counts,
        "artist_play_counts": artist_play_counts,
        "artist_queued_counts": artist_queued_counts,
        "total_queued": len(queued),
        "total_played": len(played),
        "skip_recovery_armed": skip_streak >= 2,
        "affinity": affinity,
        "completes": completes,
        "last_play": last_play,
        "n_session_tracks": n_rows,
    }


# --- Maintenance: abandon idle sessions, purge expired ones ------------------

def cleanup(now: datetime | None = None, db_path: Path | None = None) -> dict[str, int]:
    """Session TTL maintenance (reaper task / CLI / on demand).

    1. Active sessions older than SESSION_MAX_AGE_HOURS **that cannot prove
       liveness** -> resolve any in-flight track and mark 'expired'. See
       session_is_live: a fresh event heartbeat AND real playback progress,
       the latter on a deadline scaled to what is playing. Age alone never
       expires a session -- a continuously playing output stays active
       indefinitely, however long it runs.
    2. Active sessions idle past IDLE_ABANDON_MINUTES -> resolve any in-flight
       track (stored ratio, else partial/unknown -- never assumed played) and
       mark the session 'abandoned'. PAUSED sessions are exempt: a pause is an
       intentional silence, governed by the customer resume grace (4.), not by
       the idle window or the liveness watchdog.
    3. Sessions in any terminal status more than SESSION_TTL_DAYS ago -> purge
       rows (sessions, tracks, events, constraints). sessions.db is operational
       state, not history; permanent facts already left via play_events.
    4. Paused sessions whose resume grace has expired (settings:
       playback.resume_grace_minutes) -> resolve the in-flight track at the
       FROZEN pause checkpoint with the neutral 'pause_timeout' reason and
       mark the session 'abandoned'. null = NEVER: no time-based close at
       all -- only positive lifecycle evidence (stop/off, replacement,
       supersede) ends a paused attachment. No playback is ever manufactured
       during a pause, and the timeout itself is never a user verdict on the
       track.
    """
    ensure_sessions_schema(db_path)
    now_dt = now or service_clock.legacy_now()
    # Every cutoff here destroys (expire, abandon, purge): DST-safe,
    # never early, at most one repeated hour late.
    age_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(hours=SESSION_MAX_AGE_HOURS))
    idle_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(minutes=IDLE_ABANDON_MINUTES))
    purge_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(days=SESSION_TTL_DAYS))
    stats = {"expired": 0, "abandoned": 0, "purged": 0, "pause_timeout": 0,
             "dj_revoked": 0, "dj_retired": {"rows": 0, "files": 0,
                                             "orphans": 0}}
    try:
        import dj_engine
        stats["dj_revoked"] = dj_engine.revoke_stale(now_dt, db_path)
        stats["dj_retired"] = dj_engine.retire_expired(now_dt, db_path)
        # The retired-token ledger is bounded on the same sweep that fills
        # it. Age first, then the row cap; both far longer than any Music
        # Assistant queue reference, so this normally deletes nothing and
        # writes nothing (see program_tokens and STORAGE_ENDURANCE).
        import program_tokens
        # Once per installation in practice: the writes are INSERT OR
        # IGNORE, so after the first sweep this finds nothing new to say.
        try:
            import news_schedule
            stats["news_spent_backfilled"] = (
                news_schedule.backfill_spent_tokens(db_path, now_dt))
        except Exception:  # noqa: BLE001 - never breaks the sweep
            stats["news_spent_backfilled"] = 0
        stats["retired_tokens_pruned"] = program_tokens.prune(now_dt, db_path)
    except Exception:
        pass  # DJ maintenance must never take session cleanup down

    c = _connect(db_path)
    try:
        # Liveness is decided in Python, not SQL: the progress deadline
        # depends on each session's current track duration, and the row count
        # here is a handful. Doing it in SQL would also mean mixing SQLite's
        # datetime() output (space-separated) with our 'T'-separated ISO
        # strings, which compare wrongly as text. Paused sessions are exempt:
        # a pause emits no events and no progress BY DESIGN.
        candidates = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions "
            "WHERE status = 'active' AND paused_at IS NULL "
            "AND started_at < ?", (age_cutoff,),
        ).fetchall()
        overage = [row for row in candidates
                   if not session_is_live(_session_dict(row), now_dt)]
    finally:
        c.close()
    for row in overage:
        session = _session_dict(row)
        c = _connect(db_path)
        try:
            # Max-age expiry is the same class as idle abandonment for the
            # in-flight track: the SYSTEM closed the session, the user did not
            # judge the track. Both cleanup paths tag session_abandoned.
            _resolve_current(c, session, ratio=_effective_ratio(session),
                             explicit_skip=False, source="inferred",
                             reason="session_abandoned", now=now_dt)
            c.commit()
        finally:
            c.close()
        if end_session(session["session_id"], status="expired", now=now_dt,
                       db_path=db_path):
            stats["expired"] += 1

    c = _connect(db_path)
    try:
        idle = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions "
            "WHERE status = 'active' AND paused_at IS NULL "
            "AND last_event_at < ?", (idle_cutoff,),
        ).fetchall()
    finally:
        c.close()
    for row in idle:
        session = _session_dict(row)
        c = _connect(db_path)
        try:
            _resolve_current(c, session, ratio=_effective_ratio(session),
                             explicit_skip=False, source="inferred",
                             reason="session_abandoned", now=now_dt)
            c.commit()
        finally:
            c.close()
        if end_session(session["session_id"], status="abandoned", now=now_dt,
                       db_path=db_path):
            stats["abandoned"] += 1

    # 4. Pause-grace expiry. Grace is read fresh from settings each pass so a
    # customer change applies to already-paused sessions without a restart.
    # null = Never: this whole step is skipped, and a paused session ends only
    # on positive lifecycle evidence.
    grace = resume_grace_minutes()
    if grace is None:
        return stats
    grace_cutoff = service_clock.cutoff_destructive(
        now_dt, timedelta(minutes=grace))
    c = _connect(db_path)
    try:
        paused_rows = c.execute(
            f"SELECT {', '.join(_SESSION_COLS)} FROM sessions "
            "WHERE status = 'active' AND paused_at IS NOT NULL "
            "AND paused_at < ?", (grace_cutoff,),
        ).fetchall()
    finally:
        c.close()
    for row in paused_rows:
        session = _session_dict(row)
        paused_at = session["paused_at"]
        c = _connect(db_path)
        try:
            _resolve_current(c, session, ratio=_pause_ratio(session),
                             explicit_skip=False, source="inferred",
                             reason="pause_timeout", now=now_dt)
            c.execute(
                "INSERT INTO session_events (session_id, at, kind, payload) "
                "VALUES (?, ?, 'pause_timeout', ?)",
                (session["session_id"], _iso(now_dt),
                 json.dumps({"paused_at": paused_at,
                             "grace_minutes": grace,
                             "pause_position": session.get("pause_position")})))
            c.commit()
        finally:
            c.close()
        if end_session(session["session_id"], status="abandoned", now=now_dt,
                       db_path=db_path):
            stats["pause_timeout"] += 1

    c = _connect(db_path)
    try:
        expired = [r[0] for r in c.execute(
            "SELECT session_id FROM sessions WHERE status != 'active' AND ended_at < ?",
            (purge_cutoff,),
        ).fetchall()]
        for sid in expired:
            c.execute("DELETE FROM session_tracks WHERE session_id = ?", (sid,))
            c.execute("DELETE FROM session_events WHERE session_id = ?", (sid,))
            c.execute("DELETE FROM session_constraints WHERE session_id = ?", (sid,))
            c.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        c.commit()
        stats["purged"] = len(expired)
    finally:
        c.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="playback", description="Playback Intelligence")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cleanup", help="Abandon idle sessions and purge expired ones")
    sub.add_parser("features", help="Rebuild pi_track_features (derived, disposable)")
    args = parser.parse_args(argv)
    if args.command == "cleanup":
        stats = cleanup()
        print(f"cleanup: expired {stats['expired']}, abandoned {stats['abandoned']}, "
              f"purged {stats['purged']} (age>{SESSION_MAX_AGE_HOURS:g}h without "
              f"lease<{SESSION_LEASE_MINUTES:g}min + progress within "
              f"{SESSION_PROGRESS_MIN_HOURS:g}-{SESSION_PROGRESS_MAX_HOURS:g}h, "
              f"idle>{IDLE_ABANDON_MINUTES:g}min, ttl>{SESSION_TTL_DAYS:g}d)")
    elif args.command == "features":
        stats = build_features()
        print(f"features: {stats['tracks']} tracks, genre known {stats['genre_known_pct']}%, "
              f"decade known {stats['decade_known_pct']}%, "
              f"mean confidence {stats['mean_confidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
