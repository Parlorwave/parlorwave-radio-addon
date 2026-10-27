"""Durable knowledge that a program token has been retired.

WHY THIS EXISTS, in one sentence: a Music Assistant queue item is
addressable forever, and the rows that describe our program audio are
deliberately not.

`program_audio` states the transport contract -- a KNOWN program token must
never be transport-fatal -- and `dj_engine.audio_available` /
`news_bulletins.audio_available` implement it by looking the token up and
serving a short silence when it is no longer live. Both of those lookups
ask a bounded table. `dj_engine.retire_expired` deletes terminal dj_breaks
rows after 24 hours (6 for other terminal states, and a 200-row cap);
`news_bulletins.cleanup` prunes non-cached rows past ROW_CAP. The moment a
row goes, its token stops being KNOWN and starts being UNKNOWN -- and
unknown is an honest 404, which is exactly the transport-fatal answer the
contract forbids.

That gap is real, not imagined. Aired DJ tokens
sitting ahead of the play head in a Music Assistant queue -- put
back there by a restart rewind -- answer **404** once their rows have
aged out while the queue entries have not. It is the same failure mode
program_audio.py was written for, arriving by
a different door.

WHAT IS STORED, and it is the whole row: the token, which KIND of program
audio it was, and WHEN it was retired. Nothing else. No spoken text, no
file paths, no telemetry, no titles, no customer content of any kind. This
table answers one question -- "has this token been retired?" -- and if it
ever grows a fourth meaningful column, that is the moment to ask whether a
second history is being built by accident.

WHY IT CAN BE SMALL AND STILL BE ENOUGH. A retirement fact is ~60 bytes. A
station retires on the order of forty breaks a day, so the bounds below
(90 days, 20 000 rows) cost about a megabyte at the very most and represent
many months -- against Music Assistant queues that can gather a hundred or
more program references in a couple of days. The margin is wide, and, unlike keeping the full
rows forever, it is BOUNDED.

WHAT UPGRADING CANNOT RECOVER, stated plainly because the alternative is
pretending otherwise.

News migrates cleanly: `news_schedule.backfill_spent_tokens` replays the
`aired_tokens` ring and `last_aired_token`, both of which mean exactly "this
token reached air", so an installation arrives here knowing what it already
knew.

DJ BREAKS DO NOT, and cannot. A break's row WAS its only record; once
`retire_expired` deleted it, nothing anywhere states that the token ever
existed. There is no ring, no history file and no second copy to replay. So
a DJ token whose row was already reaped before this table existed stays
unknown, and its Music Assistant queue item keeps answering 404 -- the
behaviour it had before, unchanged, for exactly as long as that queue item
survives.

The one place those tokens are still written down is the Music Assistant
queue itself, and harvesting them from there is deliberately NOT done. A
32-hex string inside a URL shaped like ours is not proof that we issued it;
treating it as proof would mean minting retirement facts for anything that
can put a plausible URL in a queue, which is precisely the "resilience for
our own history, never for probes" line that keeps the 404 honest. A
truthful one-time limitation is worth more than a backfill built on a guess.

WHAT THIS IS NOT. It is not a record of what the station said, it is not
consulted for deduplication or scheduling, and nothing here may ever become
an input to what goes on the air. It has three readers: the audio-serving
gate; queue hygiene asking the same "was this ours, and is it over" question
before it will remove a queue item (retired_kind); and the news planner,
which REFUSES a retired bulletin (news_schedule._bulletin_retired) -- the
one way it can affect planning is to stop a token whose URL already answers
silence from being handed to Music Assistant as news.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

logger = logging.getLogger("musicbrain")

#: What a token was, so a future reader can tell the two program paths
#: apart without joining anything.
KIND_DJ = "dj"
KIND_NEWS = "news"
KINDS = (KIND_DJ, KIND_NEWS)

#: RETIREMENT IS NOT FORGOTTEN ON A TIMER. Expiring facts at, say, 90 days
#: would quietly reintroduce the exact defect the table exists to close: on
#: day 91 a known retired token becomes an unknown one again and the media
#: endpoint goes back to 404.
#:
#: To prune safely we would have to prove a maximum lifetime for a Music
#: Assistant queue reference, and there is no such proof. A queue can
#: hold hundreds of items spanning days; a listener who never clears a queue holds
#: them indefinitely; a restart can rewind into any of them. An unproven
#: lifetime is not a basis for deleting the fact that makes them safe.
#:
#: The cost of never forgetting is small: about 133 bytes per row with both
#: indexes, so at, say, sixty retirements a day (DJ breaks plus hourly news)
#: it is about 3 MB a year, 30 MB a decade. That is not worth trading transport
#: safety for.
RETAIN_DAYS = None        # no age-based deletion, deliberately

#: NOT A RETENTION POLICY -- a safety valve against a malfunction.
#:
#: Normal operation reaches five million rows in roughly two centuries, so
#: this can only be hit by a defect that retires in a loop. It is here so
#: that such a defect costs bounded disk (~660 MB) instead of a full
#: filesystem, and for no other reason. If it ever trips, the bug is the
#: retirement loop, not the table.
ROW_CAP = 5_000_000


def _connect(db_path=None):
    import playback
    playback.ensure_sessions_schema(db_path)
    return playback._connect(db_path)


def _iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def record(conn, tokens: Iterable[str], kind: str,
           now: datetime | None = None) -> int:
    """Write retirement facts on an OPEN connection, without committing.

    Takes the connection rather than opening its own on purpose: the DJ
    reaper must be able to write the fact and delete the row in ONE durable
    operation, so that no crash can leave a token whose row is gone and
    whose retirement was never recorded. That ordering is the entire point
    of this function's shape.

    Idempotent by primary key: retiring the same token twice keeps the
    FIRST timestamp, because the first one is when it actually stopped being
    live and re-stamping it would quietly extend its life in the table.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown program token kind: {kind!r}")
    stamp = _iso(now)
    n = 0
    for token in tokens:
        if not token:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO retired_program_tokens "
            "(token, kind, retired_at) VALUES (?, ?, ?)", (token, kind, stamp))
        n += cur.rowcount
    return n


def retire(tokens: Iterable[str], kind: str, now: datetime | None = None,
           db_path=None) -> int:
    """Record retirement in its own transaction, for callers that are not
    already inside one. Best-effort and never raises: failing to write a
    tombstone fact must not be able to abort the operation that caused it."""
    try:
        c = _connect(db_path)
        try:
            n = record(c, tokens, kind, now)
            c.commit()
            return n
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - transport safety is best-effort
        logger.warning("retired-token record failed (transport safety only)",
                       exc_info=True)
        return 0


def is_retired(token: str, db_path=None) -> bool:
    """Has this token been retired? Fails CLOSED to 'no'.

    'No' means the caller falls through to its existing behaviour -- a 404
    for an unknown token -- which is what happened before this table
    existed. An unreadable ledger therefore degrades to the old behaviour
    rather than to serving something, and never to an exception inside a
    media request."""
    if not token:
        return False
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT 1 FROM retired_program_tokens "
                            "WHERE token = ?", (token,)).fetchone()
        finally:
            c.close()
        return row is not None
    except Exception:  # noqa: BLE001
        logger.warning("retired-token lookup failed; treating as unknown",
                       exc_info=True)
        return False


def retired_kind(token: str, db_path=None) -> str | None:
    """Which KIND of program audio this token was, if it has been retired;
    None when it has not (or the ledger cannot be read -- fails CLOSED to
    'not known to be retired', like is_retired).

    The second reader of this table, and a deliberately narrow one: queue
    hygiene must not delete a queue item on the strength of a URL that
    merely looks like ours, so it asks whether Parlorwave itself recorded
    the token's retirement -- and of which kind, so a break token can never
    be vouched for by a news fact or the reverse. Nothing about what goes
    on the air is decided from this."""
    if not token:
        return None
    try:
        c = _connect(db_path)
        try:
            row = c.execute("SELECT kind FROM retired_program_tokens "
                            "WHERE token = ?", (token,)).fetchone()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        logger.warning("retired-token lookup failed; treating as unknown",
                       exc_info=True)
        return None
    if row is None:
        return None
    kind = str(row[0] or "")
    return kind if kind in KINDS else None


def prune(now: datetime | None = None, db_path=None) -> int:
    """Enforce the emergency cap only. Normally deletes nothing, ever.

    There is deliberately no age pass: see RETAIN_DAYS. Age-based deletion
    would turn a known retired token back into an unknown one, which is the
    404 this whole table exists to prevent, and nothing proves a Music
    Assistant queue reference cannot outlive any horizon we might pick.

    `RETAIN_DAYS` is honoured if an operator sets it, so the mechanism is
    still there for a deployment that can prove a bound. It is None by
    default because we cannot.

    Returns the number of facts forgotten -- which on a healthy
    installation is 0 on every call for the life of the product."""
    now_dt = now or datetime.now()
    try:
        c = _connect(db_path)
        try:
            gone = 0
            if RETAIN_DAYS is not None:
                cutoff = _iso(now_dt - timedelta(days=float(RETAIN_DAYS)))
                gone += c.execute(
                    "DELETE FROM retired_program_tokens WHERE retired_at < ?",
                    (cutoff,)).rowcount
            gone += c.execute(
                "DELETE FROM retired_program_tokens WHERE token IN ("
                "  SELECT token FROM retired_program_tokens "
                "  ORDER BY retired_at DESC, token LIMIT -1 OFFSET ?)",
                (ROW_CAP,)).rowcount
            c.commit()
            return gone
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        logger.warning("retired-token prune failed", exc_info=True)
        return 0


def count(db_path=None) -> int:
    """How many facts are held. Diagnostics only."""
    try:
        c = _connect(db_path)
        try:
            return c.execute(
                "SELECT COUNT(*) FROM retired_program_tokens").fetchone()[0]
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return 0
