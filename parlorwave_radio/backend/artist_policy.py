"""Back-to-back same-artist programming policy.

A customer taste control over how often ParlorWave's AUTONOMOUS station runway
places two tracks by the same primary artist next to each other. It
governs ONLY the batch ParlorWave itself is programming (the station refill);
manual additions, adopted queues, and intentional album/artist playback
are the listener's and are never touched by this module.

It is the DOWNSTREAM DETERMINISTIC enforcement in the station-ordering pipeline: the LLM
may propose an order, but this validator/repair has the final say. It
always returns a PERMUTATION of the exact candidates -- never drops, adds,
replaces, or invents a track.

Levels (deterministic, evaluated over the actual batch, not per-track coin
flips). Let n = music candidates; an "adjacency budget" caps how many
same-artist adjacent PAIRS the repair will deliberately create:

    off     budget 0   -- spread artists apart; only unavoidable pairs remain
    low     budget n//10 (rare doubles)
    medium  budget n//5  (occasional doubles)
    high    budget n     (doubles common)

HARD invariant, every level: a same-artist RUN longer than 2 is avoided
whenever the pool allows it. A run of 3+ only survives when the candidate
composition makes it unavoidable (the minimum-feasible case, e.g. the pool
is almost entirely one artist), never as ordinary programming. "High"
means doubles are common, NOT six-in-a-row.

Repair strategy:
  * Fast path -- if the incoming order already conforms (max run <= 2 and
    no more same-artist pairs than the budget allows) it is returned
    UNCHANGED, so a well-spread programmer order keeps its discovery cadence. High
    is the one exception: it actively favours doubles, so it always runs
    the builder.
  * Otherwise a single deterministic greedy rebuild: walk position by
    position, repeat the last artist to spend the pair budget (never past a
    run of 2), and when it must switch it brings forward the DIFFERENT
    artist with the most remaining tracks (ties broken by earliest original
    position). Preferring the heaviest remaining artist is what stops one
    artist being stranded into a long tail run -- the bug a naive
    order-preserving pass hits. Only when a single artist is all that
    remains is an unavoidable pair (or >2 run) emitted: minimum feasible.

Artist identity: the track's stable MusicBrainz artist id when the
candidate carries one (`mb_artist_id`), else its canonical artist string
casefolded. No fuzzy matching; a collaboration is identified by its single
stored artist field (the primary artist as the catalog records it).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

VALID_FREQ = ("low", "medium", "high")


def _budget(n: int, enabled: bool, frequency: str) -> int:
    if not enabled:
        return 0
    if frequency == "low":
        return n // 10
    if frequency == "high":
        return n
    return n // 5            # medium (and any unexpected value) default


def primary_artist(track: dict[str, Any]) -> str:
    """Stable primary-artist identity for adjacency (never sort_index, never
    fuzzy). MBID when present, else casefolded artist string."""
    mbid = track.get("mb_artist_id")
    if isinstance(mbid, str) and mbid.strip():
        return f"mb:{mbid.strip()}"
    return "name:" + str(track.get("artist") or "").strip().lower()


def _adjacent_pairs(order: list[dict[str, Any]], key: Callable = primary_artist) -> int:
    keys = [key(t) for t in order]
    return sum(1 for a, b in zip(keys, keys[1:]) if a == b)


def _max_run(order: list[dict[str, Any]], key: Callable = primary_artist) -> int:
    best = run = 0
    prev = None
    for t in order:
        k = key(t)
        run = run + 1 if k == prev else 1
        best = max(best, run)
        prev = k
    return best


def _pick_switch(remaining: list[dict[str, Any]], key: Callable,
                 last: str | None, counts: Counter) -> int | None:
    """Index in `remaining` of the next track to place when switching away
    from `last`: the different-artist track whose artist has the most
    remaining copies (earliest original position breaks ties). Returns None
    when every remaining track is `last`'s artist."""
    best_idx = None
    best_count = -1
    for i, t in enumerate(remaining):
        k = key(t)
        if k == last:
            continue
        c = counts[k]
        if c > best_count:          # strict: earliest index wins a tie
            best_count = c
            best_idx = i
    return best_idx


def _greedy(tracks: list[dict[str, Any]], key: Callable,
            budget: int) -> tuple[list[dict[str, Any]], int]:
    """Deterministic rebuild. Returns (order, forced_adjacencies)."""
    remaining = list(tracks)
    out: list[dict[str, Any]] = []
    pairs_used = 0
    forced = 0
    counts: Counter = Counter(key(t) for t in remaining)
    while remaining:
        last = key(out[-1]) if out else None
        run2 = len(out) >= 2 and key(out[-2]) == key(out[-1])
        can_repeat = (last is not None and not run2
                      and pairs_used < budget and counts[last] > 0)
        if can_repeat:
            idx = next(i for i, t in enumerate(remaining) if key(t) == last)
            pairs_used += 1
        else:
            idx = _pick_switch(remaining, key, last, counts)
            if idx is None:
                # Only `last`'s artist remains: an adjacency (and past a run
                # of 2, a >2 run) is unavoidable. Minimum feasible.
                idx = 0
                if last is not None:
                    forced += 1
        chosen = remaining.pop(idx)
        counts[key(chosen)] -= 1
        out.append(chosen)
    return out, forced


def enforce(tracks: list[dict[str, Any]], *, enabled: bool = True,
            frequency: str = "medium",
            identity: Callable[[dict[str, Any]], str] | None = None
            ) -> dict[str, Any]:
    """Return {"order": [tracks permuted], "telemetry": {...}}. The order is
    always an exact permutation of `tracks`. Never raises for ordinary
    input; a degenerate batch (< 3) is returned unchanged."""
    key = identity or primary_artist
    n = len(tracks)
    budget = _budget(n, enabled, frequency)
    tel: dict[str, Any] = {
        "enabled": enabled, "frequency": frequency if enabled else None,
        "candidate_pairs": _adjacent_pairs(tracks, key),
        "budget": budget,
        "repaired": False, "final_pairs": 0, "max_run": 0,
        "perfect_impossible": False}
    if n < 3:
        tel["final_pairs"] = _adjacent_pairs(tracks, key)
        tel["max_run"] = _max_run(tracks, key)
        return {"order": list(tracks), "telemetry": tel}

    in_pairs = tel["candidate_pairs"]
    in_run = _max_run(tracks, key)
    # Fast path: a conforming order is left alone (minimal movement). High
    # actively favours doubles, so it always rebuilds.
    conforms = in_run <= 2 and in_pairs <= budget
    if conforms and not (enabled and frequency == "high"):
        order, forced = list(tracks), 0
    else:
        order, forced = _greedy(tracks, key, budget)

    tel["final_pairs"] = _adjacent_pairs(order, key)
    tel["max_run"] = _max_run(order, key)
    tel["repaired"] = [key(t) for t in order] != [key(t) for t in tracks]
    tel["perfect_impossible"] = (
        tel["max_run"] > 2 or (not enabled and tel["final_pairs"] > 0))
    if forced:
        tel["forced_adjacencies"] = forced
    return {"order": order, "telemetry": tel}
