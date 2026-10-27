#!/usr/bin/env python3
"""The ONE serialization point for station-owned Music Assistant queue writes.

WHY GLOBAL AND NOT PER-QUEUE. The obvious design is a lock per queue_id, and
it is wrong here, because Start Radio reaches ACROSS outputs: an explicit
request on queue B treats paused/idle sessions on queue A as handoff
candidates and retires them. A lock held on B excludes nothing on A, so a
station-owned writer that was mid-flight for the old station -- selection and
URI resolution take seconds, since resolution is one serial MA search per pick
at MA_TIMEOUT_S each -- could still write to A after the listener has already
moved the station to B.

A single global lock also states the v1 product invariant exactly: at most one
active station exists, so at most one station-owned queue mutation may be in
flight. A per-queue lock would model a multi-station world that does not
exist.

WHAT IT COVERS. Station-owned FUTURE mutations only:

    Start Radio        ma_queue.replace_future
    station refill     ma_queue.append_tracks   (option:"add")

WHAT IT DELIBERATELY DOES NOT COVER. Ordinary DJ break insertion and the news
intro/bulletin pair, both of which write the NEXT position rather than the
tail. Pulling them in would let a slow station rebuild delay a
boundary-critical break or split the atomic news pair, and they do not race
with station ownership: a NEXT insertion adds items in the MIDDLE of the
queue, never a contiguous run at the end, so refill reconciliation tolerates
it by construction.

DEPLOYMENT REALITY THIS RELIES ON. `uvicorn --workers 1` in the Dockerfile,
the compose file and the running container: ONE process, with concurrency
provided by threads (the observer worker, DJ and news passes, ack renders).
So a process-local threading.Lock genuinely excludes every writer -- the same
primitive dj_engine._news_worker, news_schedule._pass_lock and
ma_observer._worker_lock already use. An asyncio.Lock would NOT do: the
writers are threads, not coroutines.

WHAT IT CANNOT DO. A lock dies with the process. Crash safety comes from the
durable `preparing` station row and the `pending` refill batch, which are
written BEFORE the external call precisely so a restart can prove what
happened by reading the queue back. The lock protects live concurrency; the
rows protect crashes. Neither substitutes for the other.

NEVER HOLD A SQLITE TRANSACTION WHILE HOLDING THIS. The whole point is that
the network call happens inside the lock and outside any transaction.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

# Module-level and process-global on purpose: one station, one writer.
_station_mutation_lock = threading.Lock()

# Long enough to cover revalidation + one MA write + one readback at
# MA_TIMEOUT_S=5 apiece with room to spare; short enough that a wedged holder
# surfaces as a refused mutation rather than a hung request thread. Refusal is
# always safe here: the caller's contract is "no station change", never a
# retry or a partial write.
ACQUIRE_TIMEOUT_S = 30.0


class StationMutationBusy(RuntimeError):
    """Another station-owned queue mutation is in flight. The caller must
    treat this as 'no station change happened' -- never as a reason to write
    to Music Assistant anyway."""


@contextmanager
def station_mutation(timeout: float | None = None):
    """Serialize one station-owned MA queue mutation.

    Raises StationMutationBusy rather than blocking forever, so a wedged
    holder can never turn into a wedged request thread."""
    wait = ACQUIRE_TIMEOUT_S if timeout is None else timeout
    if not _station_mutation_lock.acquire(timeout=wait):
        raise StationMutationBusy(
            f"station mutation lock not acquired within {wait:g}s")
    try:
        yield
    finally:
        _station_mutation_lock.release()


def is_held() -> bool:
    """Diagnostics only (status surfaces, tests). Never a gate: a lock that is
    free right now says nothing about whether it will still be free a moment
    later, which is exactly what acquiring it is for."""
    return _station_mutation_lock.locked()
