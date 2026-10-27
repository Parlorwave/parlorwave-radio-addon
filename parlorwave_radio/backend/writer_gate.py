"""Who may use the writer model right now: live work always, background story
preparation only when nothing live wants it.

Live writer demand is SIGNALLED, never queued. Every in-process call to the
writer runtime goes through providers.ollama_generate / ollama_warm, which
wrap themselves in `live()`: a counter goes up, the epoch moves, and the call
proceeds at once. Nothing live ever waits here -- the lock below is held for
an increment, never across I/O, and background preparation never holds it
while it talks to the writer. (ANNOUNCE NOW and every future immediate path
inherit this for free: they cannot depend on the preparation lock, because
the only thing on their path is this counter.)

Background preparation asks two things:

  * `quiet()` before it starts a call: no live call in flight, and none
    ended within LIVE_COOLDOWN_S -- a counter, not a flag, so two
    overlapping live requests keep it out until BOTH are done, and the
    cool-down covers the gap between steps of one live job (planner, then
    a free-text retry, then the next break's warm);
  * `interrupted(since_epoch)` while its call streams: true as soon as any
    live call has STARTED since it began, even one that already finished --
    so a live request can never slip in and out between two checks unseen.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Callable

LIVE_COOLDOWN_S = 10.0

_lock = threading.Lock()
_in_flight = 0
_epoch = 0
_last_end = float("-inf")
_clock: Callable[[], float] = time.monotonic


@contextlib.contextmanager
def live():
    """Mark one live writer call for its duration. Never blocks on anything
    background work holds."""
    global _in_flight, _epoch, _last_end
    with _lock:
        _in_flight += 1
        _epoch += 1
    try:
        yield
    finally:
        with _lock:
            _in_flight -= 1
            _last_end = _clock()


def epoch() -> int:
    with _lock:
        return _epoch


def in_flight() -> int:
    with _lock:
        return _in_flight


def quiet(cooldown_s: float = LIVE_COOLDOWN_S) -> bool:
    """May background work START a writer call now?"""
    with _lock:
        return _in_flight == 0 and _clock() - _last_end >= cooldown_s


def interrupted(since_epoch: int) -> bool:
    """Has any live call started since `since_epoch`, or is one running?"""
    with _lock:
        return _in_flight > 0 or _epoch != since_epoch


def _reset_for_tests(clock: Callable[[], float] | None = None) -> None:
    global _in_flight, _epoch, _last_end, _clock
    with _lock:
        _in_flight, _epoch, _last_end = 0, 0, float("-inf")
        _clock = clock or time.monotonic
