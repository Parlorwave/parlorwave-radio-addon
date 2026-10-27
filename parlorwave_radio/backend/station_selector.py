#!/usr/bin/env python3
"""Choosing the next tracks for a station, without asking the LLM again.

A station's frozen profile says WHICH ARTISTS belong to it and roughly how
much each one matters. That is the only thing Start Radio's LLM call is
needed for, and schema v9 already stores it, so every later refill can be
chosen here -- deterministically, offline, with no Ollama dependency, no GPU
contention and no risk that a second LLM call quietly drifts the station's
character.

PURE. No Music Assistant, no network, no database, no clock of its own, no
global random state. Everything it reasons about is handed to it: the catalog,
the current scores, what is already queued, what was played recently, what was
generated recently, and a random.Random the caller owns. That makes every
rotation rule a table test instead of an integration test.

WHAT IS FROZEN AND WHAT IS LIVE. `artist_weights` means "these artists belong
to this station, in roughly this proportion" -- it does NOT mean "ignore what
the listener does next". Eligibility, scoring, thumbs, skips and catalog
membership are all read LIVE at selection time through the same `allowed()`
and `effective_score()` the initial Start Radio uses. A frozen artist whose
tracks have all been downvoted simply stops contributing; new tracks by a
frozen artist become eligible the moment they are indexed. The profile is
never mutated to reflect any of that.

MUSIC CONTINUITY WINS. Rotation preferences relax in four deterministic
passes, and the last one keeps almost nothing. A station that cannot honour
its preferred spacing should sound slightly repetitive; it should never fall
silent. If even the floor cannot fill the batch, the partial batch is returned
truthfully rather than nothing.

WHAT THIS MODULE DOES NOT DO: resolve anything to Music Assistant URIs, decide
WHEN to refill, decide HOW MANY to ask for, touch station state, or know that
`dj.enabled` or a future keep-radio-playing setting exist. Those are the
caller's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable

# --- pass names, in relaxation order -----------------------------------------
PREFERRED = "preferred"
BROADENED = "broadened"
SHORT_ROTATION = "short_rotation"
CONTINUITY_FLOOR = "continuity_floor"


@dataclass(frozen=True)
class Pass:
    """One rung of the relaxation ladder.

    `recent_hours` / `recent_tracks` are OR-ed, deliberately: a wall clock
    alone mishandles an overnight pause (four hours pass while nothing plays,
    so everything looks fresh again), and a track count alone mishandles a
    long idle gap (twenty tracks ago might be last week). Either signal saying
    "too soon" is enough.

    The penalties are multiplicative weight factors, never bans -- an artist
    or album that just played becomes unlikely, not impossible, which is what
    keeps a two-artist station from deadlocking or degenerating into strict
    alternation.
    """
    name: str
    recent_hours: float
    recent_tracks: int
    exclude_recently_generated: bool
    generated_factor: float          # applied when not excluded outright
    artist_factor: float             # per occurrence within artist_window
    artist_window: int
    album_factor: float
    album_window: int
    top_tier_ratio: float            # per-artist score-rank cut, 1.0 = no cut
    # How hard the frozen roster weights are followed: weight ** power.
    # 1.0 = as written; below 1 flattens the roster (lower-weight artists
    # get more airtime), above 1 leans harder into its strongest artists.
    # Runtime behaviour like everything else here -- the roster is untouched.
    weight_power: float = 1.0
    # How hard PERSONAL familiarity (station_features, 0..1 per track) pulls
    # the pick. 0.0 = ignored entirely -- the original behaviour, and
    # the behaviour whenever the caller supplies no features. Positive
    # values multiply each candidate's weight by (0.25 + familiarity) **
    # power, so a well-known song outweighs an unheard one without ever
    # zeroing the unheard one out.
    familiarity_power: float = 0.0


@dataclass(frozen=True)
class RefillPolicy:
    """RUNTIME selection behaviour. Deliberately NOT part of station identity
    and never serialized into a station row: a station started months ago
    must not be pinned to that month's spacing constants, and future
    hit-awareness will land as different policy values rather than as a second
    copy of this module."""
    passes: tuple[Pass, ...]
    min_tier_tracks: int = 5
    # Anchor cadence: with N > 0, no run of N consecutive picks in a batch
    # may be all-unrecognisable -- when a run reaches N-1, the next pick is
    # restricted to RECOGNISABLE tracks (familiarity >= anchor_min) if any
    # candidate qualifies; a naturally picked anchor resets the run. 0 = off.
    # This is Discovery's safety line: explore the roster, but come back to
    # something the listener knows before they reach for the skip button.
    anchor_every: int = 0
    anchor_min: float = 0.5

    def __iter__(self):
        return iter(self.passes)


DEFAULT_POLICY = RefillPolicy(passes=(
    # 1. Ordinary radio rotation: nothing heard in the last four hours or the
    #    last forty tracks, nothing generated recently, artists and albums
    #    spaced out, and the better half of each artist's catalog preferred.
    #
    #    artist_factor 0.35 / window 3 is MEASURED, not guessed. On the worst
    #    case for weight fidelity -- a two-artist profile weighted 10:2, which
    #    is exactly where naive round-robin would collapse to 50/50 -- it
    #    yields 1.72:1 across 288 picks with never more than 4 of one artist
    #    in a row. The alternatives traded badly: 0.45/window-2 gives better
    #    fidelity (2.20:1) but runs of 6, and 0.55/window-2 gives 2.56:1 with
    #    runs of 10, which is audibly wrong on radio. Typical profiles carry
    #    5-12 artists, where the same constants produce runs of 2-3.
    Pass(PREFERRED, 4.0, 40, True, 1.0, 0.35, 3, 0.25, 4, 0.6),
    # 2. Same rotation window, but stop insisting on album spacing and on the
    #    top score tier -- the cheapest constraints to give up first, because
    #    they shape polish rather than repetition.
    Pass(BROADENED, 4.0, 40, True, 1.0, 0.50, 3, 1.0, 1, 1.0),
    # 3. Shorter rotation and looser artist spacing. Recently-generated
    #    material stops being excluded and becomes merely unattractive: a
    #    track the listener removed from the queue should not come straight
    #    back, but it must not be banished either.
    Pass(SHORT_ROTATION, 1.0, 15, False, 0.30, 0.70, 2, 1.0, 1, 1.0),
    # 4. The floor. Only the hard rules plus protection against an almost
    #    immediate repeat. Past here, silence would be the alternative -- so
    #    if even this window would empty the pool, it COLLAPSES to protecting
    #    just the track that played last (see _collapsed). A station with
    #    fifteen songs, all heard in the past twenty minutes, still gets
    #    music; it simply gets repetitive music, which is the correct trade.
    Pass(CONTINUITY_FLOOR, 0.34, 5, False, 1.0, 1.0, 1, 1.0, 1, 1.0),
))


def _collapsed(rung: Pass) -> Pass:
    """The floor with its rotation window reduced to the single most recent
    track. Reached only when the ordinary floor would return NOTHING while
    hard-valid tracks still exist: continuity wins over cooldown purity, but
    playing the same song twice in a row never does."""
    return Pass(rung.name, 0.0, 1, False, 1.0, 1.0, 1, 1.0, 1, 1.0)


@dataclass
class RefillSelection:
    """The batch plus enough diagnosis to explain it in a log line."""
    tracks: list[dict[str, Any]] = field(default_factory=list)
    requested_count: int = 0
    returned_count: int = 0
    relaxation_pass: str | None = None      # deepest rung actually needed
    passes_used: list[str] = field(default_factory=list)
    eligible_artist_count: int = 0
    hard_excluded_count: int = 0
    recent_excluded_count: int = 0
    generated_excluded_count: int = 0
    anchor_ids: list[str] = field(default_factory=list)   # recognisable picks
    exhausted: bool = False                 # even the floor could not fill it

    def as_dict(self) -> dict[str, Any]:
        return {"tracks": self.tracks, "requested_count": self.requested_count,
                "returned_count": self.returned_count,
                "relaxation_pass": self.relaxation_pass,
                "passes_used": list(self.passes_used),
                "eligible_artist_count": self.eligible_artist_count,
                "hard_excluded_count": self.hard_excluded_count,
                "recent_excluded_count": self.recent_excluded_count,
                "generated_excluded_count": self.generated_excluded_count,
                "anchor_count": len(self.anchor_ids),
                "exhausted": self.exhausted}


# --- identity -----------------------------------------------------------------

def track_identity(track: dict[str, Any]) -> str:
    """The canonical identity of a catalog track: its PATH.

    Not artist+title. The library genuinely holds several rows sharing one
    title and artist -- "Lost Bells" may appear several times -- and a
    remaster, a live cut and an album version are different recordings that a
    metadata key would silently merge. `path` is also exactly what
    `allowed()` and `effective_score()` key their score lookups on, so using
    anything else here would mean two different notions of "the same track"
    inside one selection."""
    return str(track.get("path") or "")


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


# --- recency ------------------------------------------------------------------

def _parse_when(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def recent_track_ids(recent_played: Iterable[dict[str, Any]], *,
                     now: datetime, hours: float, count: int) -> set[str]:
    """Identities that count as "heard too recently" for one pass.

    EITHER condition is enough, and each covers the other's blind spot: the
    clock alone would call everything fresh after an overnight pause, while
    the count alone would call something fresh that was played twenty tracks
    and one week ago. `recent_played` is newest-first."""
    cutoff = now - timedelta(hours=hours)
    out: set[str] = set()
    for index, row in enumerate(recent_played or ()):
        ident = str(row.get("path") or "")
        if not ident:
            continue
        if index < count:
            out.add(ident)
            continue
        when = _parse_when(row.get("played_at"))
        if when is not None and when >= cutoff:
            out.add(ident)
    return out


def _recent_generated_ids(recently_generated: Iterable[dict[str, Any]]
                          ) -> set[str]:
    """Tracks this station put in the queue recently, whether or not they ever
    played.

    This exists for a case nothing else covers: the listener deletes a future
    station track. It never played, so it is absent from play history; it is
    no longer queued, so it is absent from the queue exclusions; and without
    this ring the very next refill could hand it straight back. It is a
    ROTATION signal, not a verdict -- we cannot distinguish "I don't want
    this" from "not right now", and the schema-lock decision was to never
    infer more intent than the evidence supports."""
    out: set[str] = set()
    for row in recently_generated or ():
        ident = str((row or {}).get("path") or "")
        if ident:
            out.add(ident)
    return out


# --- weighting ----------------------------------------------------------------

def _occurrences(sequence: list[str], value: str, window: int) -> int:
    if window <= 0 or not value:
        return 0
    return sequence[-window:].count(value)


def _spacing_factor(sequence: list[str], value: str, window: int,
                    factor: float) -> float:
    """Multiplicative discouragement, compounding per recent occurrence.

    This is deliberately NOT round-robin. Strict alternation would flatten a
    10-vs-2 artist profile into 50/50 and throw away the very thing the LLM
    was asked for. Compounding a factor instead means one repeat is mildly
    discouraged, two is strongly discouraged and three is nearly excluded,
    while a heavily weighted artist still wins most of the time -- clumping is
    controlled without the weights being overruled."""
    if factor >= 1.0:
        return 1.0
    hits = _occurrences(sequence, value, window)
    return factor ** hits if hits else 1.0


def _weighted_pick(rng, candidates: list[tuple[Any, float]]) -> Any:
    """One weighted draw from the caller's RNG. Never touches global random."""
    total = math.fsum(w for _, w in candidates)
    if total <= 0:
        return candidates[0][0] if candidates else None
    target = rng.random() * total
    running = 0.0
    for item, weight in candidates:
        running += weight
        if target <= running:
            return item
    return candidates[-1][0]


# --- selection ----------------------------------------------------------------

def _tier_cut(tracks: list[dict[str, Any]], scores: dict, ratio: float,
              minimum: int, score_fn: Callable) -> list[dict[str, Any]]:
    """Keep an artist's better material in the preferred pass.

    A quality preference, not a quality gate: it is the first thing relaxation
    gives up, and `minimum` stops a thinly-represented artist from being cut
    down to nothing."""
    if ratio >= 1.0 or len(tracks) <= minimum:
        return tracks
    ranked = sorted(tracks, key=lambda t: (-score_fn(t, scores),
                                           track_identity(t)))
    keep = max(minimum, int(len(ranked) * ratio))
    return ranked[:keep]


def select_station_refill(*, artist_weights: dict[str, int | float],
                          request_text: str,
                          catalog_tracks: list[dict[str, Any]],
                          scores: dict[str, Any],
                          rng,
                          count: int,
                          now: datetime,
                          hard_excluded_ids: Iterable[str] = (),
                          recent_played: Iterable[dict[str, Any]] = (),
                          recently_generated: Iterable[dict[str, Any]] = (),
                          policy: RefillPolicy = DEFAULT_POLICY,
                          features: dict[str, Any] | None = None,
                          artist_caps: dict[str, int] | None = None,
                          default_artist_cap: int = 0,
                          normalize_by_artist_pool: bool = False,
                          ) -> RefillSelection:
    """Choose up to `count` catalog tracks for a station refill. PURE.

    Returns catalog track dicts, NOT Music Assistant URIs: resolution is the
    caller's job and a separate failure domain, and mixing the two would make
    every rotation rule need a network to test.

    HARD exclusions hold through every pass and are never relaxed: whatever
    the caller says is already in the queue (including the current item),
    anything `allowed()` refuses under live policy, and any track already
    chosen in this batch. SOFT rotation -- recency, recently-generated,
    artist and album spacing, score tier -- relaxes across four passes, each
    topping up what the previous one left short, so the result is the
    strictest batch that could actually be filled.

    `features` (may be None) maps track identity -> an object with a
    `familiarity` attribute in 0..1 (station_features.TrackFeatures). It is
    advice, never a gate: with no features, an empty dict, or a policy whose
    familiarity_power and anchor_every are zero, selection is bit-for-bit
    what it was before features existed.

    `rng` must be a random.Random the caller owns. Nothing here reads or
    writes the global random module, so a refill can never perturb the
    initial Start Radio path's own shuffling.

    STATION PROGRAMMER OPTIONS (all off by default, so every existing
    caller selects bit-for-bit as before):

    `normalize_by_artist_pool` makes `artist_weights` mean what this
    module's doc already says it means -- a share of AIRTIME. Without it,
    every track is a candidate at its artist's weight, so an artist's
    chance grows with how many tracks it owns: on a genre station an
    artist with 300 matching tracks would outdraw one with 10 by 30x at
    equal weight. With it, each candidate's weight is divided by how many
    candidates its artist has left in the pass.

    `artist_caps` / `default_artist_cap` are HARD per-batch ceilings per
    artist (0 = none). They hold through every pass, including the floor:
    a station that cannot be filled without one artist dominating comes
    back SHORT, and the caller reports that truthfully rather than handing
    over twenty-five songs by one band."""
    from llm_station_builder import allowed, effective_score

    result = RefillSelection(requested_count=max(0, int(count)))
    if result.requested_count <= 0:
        return result

    weights = {str(a): float(w) for a, w in (artist_weights or {}).items()
               if a and float(w) > 0}
    hard = {str(x) for x in (hard_excluded_ids or ()) if x}
    recent_played = list(recent_played or ())
    generated_ids = _recent_generated_ids(recently_generated)

    # LIVE policy, applied once: the frozen roster proposes, current taste and
    # catalog state dispose.
    by_artist: dict[str, list[dict[str, Any]]] = {}
    hard_excluded = 0
    for track in catalog_tracks or ():
        artist = str(track.get("artist") or "")
        if artist not in weights:
            continue
        ident = track_identity(track)
        if not ident:
            continue
        if ident in hard:
            hard_excluded += 1
            continue
        if not allowed(track, scores, request_text or ""):
            hard_excluded += 1
            continue
        by_artist.setdefault(artist, []).append(track)

    result.hard_excluded_count = hard_excluded
    result.eligible_artist_count = len(by_artist)
    if not by_artist:
        result.exhausted = True
        return result

    def _familiarity(ident: str) -> float | None:
        entry = (features or {}).get(ident)
        value = getattr(entry, "familiarity", None)
        if value is None and isinstance(entry, dict):
            value = entry.get("familiarity")
        try:
            return None if value is None else max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _is_anchor(ident: str) -> bool:
        fam = _familiarity(ident)
        return fam is not None and fam >= policy.anchor_min

    chosen_ids: set[str] = set()
    caps = {str(a): int(c) for a, c in (artist_caps or {}).items()}
    picked_per_artist: dict[str, int] = {}

    def _capped(artist: str) -> bool:
        cap = caps.get(artist, int(default_artist_cap or 0))
        return cap > 0 and picked_per_artist.get(artist, 0) >= cap

    unanchored_run = 0        # consecutive picks with no recognisable anchor
    artist_seq: list[str] = [_norm(r.get("artist")) for r in
                             reversed(recent_played[:8])]
    album_seq: list[str] = [_norm(r.get("album")) for r in
                            reversed(recent_played[:8])]
    recent_excluded: set[str] = set()
    generated_excluded: set[str] = set()

    for rung in policy:
        if len(result.tracks) >= result.requested_count:
            break
        recent_ids = recent_track_ids(recent_played, now=now,
                                      hours=rung.recent_hours,
                                      count=rung.recent_tracks)
        pool: dict[str, list[dict[str, Any]]] = {}
        for artist, tracks in by_artist.items():
            usable = []
            for track in tracks:
                ident = track_identity(track)
                if ident in chosen_ids:
                    continue
                if ident in recent_ids:
                    recent_excluded.add(ident)
                    continue
                if rung.exclude_recently_generated and ident in generated_ids:
                    generated_excluded.add(ident)
                    continue
                usable.append(track)
            if usable:
                pool[artist] = _tier_cut(usable, scores, rung.top_tier_ratio,
                                         policy.min_tier_tracks,
                                         effective_score)
        if not pool:
            continue

        used_this_pass = False
        while len(result.tracks) < result.requested_count:
            candidates: list[tuple[dict[str, Any], float]] = []
            for artist, tracks in pool.items():
                if _capped(artist):
                    continue
                artist_factor = _spacing_factor(artist_seq, _norm(artist),
                                                rung.artist_window,
                                                rung.artist_factor)
                base = (weights.get(artist, 1.0) ** rung.weight_power
                        * artist_factor)
                if normalize_by_artist_pool:
                    left = sum(1 for t in tracks
                               if track_identity(t) not in chosen_ids)
                    base = base / left if left else 0.0
                if base <= 0:
                    continue
                for track in tracks:
                    ident = track_identity(track)
                    if ident in chosen_ids:
                        continue
                    weight = base * _spacing_factor(
                        album_seq, _norm(track.get("album")),
                        rung.album_window, rung.album_factor)
                    if ident in generated_ids and rung.generated_factor < 1.0:
                        weight *= rung.generated_factor
                    if rung.familiarity_power > 0:
                        fam = _familiarity(ident)
                        if fam is not None:
                            weight *= (0.25 + fam) ** rung.familiarity_power
                    if weight > 0:
                        candidates.append((track, weight))
            if not candidates:
                break
            if policy.anchor_every > 0 \
                    and unanchored_run >= policy.anchor_every - 1:
                # The run is about to get too long: take an anchor now if
                # any candidate qualifies. If none does (a young household,
                # a fresh roster), the cadence simply cannot be honoured
                # this pick -- advice, never a gate.
                anchors = [(t, w) for t, w in candidates
                           if _is_anchor(track_identity(t))]
                if anchors:
                    candidates = anchors
            pick = _weighted_pick(rng, candidates)
            if pick is None:
                break
            ident = track_identity(pick)
            chosen_ids.add(ident)
            result.tracks.append(pick)
            picked_per_artist[str(pick.get("artist") or "")] = \
                picked_per_artist.get(str(pick.get("artist") or ""), 0) + 1
            if _is_anchor(ident):
                result.anchor_ids.append(ident)
                unanchored_run = 0
            else:
                unanchored_run += 1
            artist_seq.append(_norm(pick.get("artist")))
            album_seq.append(_norm(pick.get("album")))
            used_this_pass = True

        if used_this_pass and rung.name not in result.passes_used:
            result.passes_used.append(rung.name)
            result.relaxation_pass = rung.name

    if not result.tracks:
        # Every rung refused. Rather than let the station fall silent, retry
        # the floor protecting only the track that just played.
        rung = _collapsed(policy.passes[-1])
        recent_ids = recent_track_ids(recent_played, now=now,
                                      hours=rung.recent_hours,
                                      count=rung.recent_tracks)
        candidates = [(t, weights.get(str(t.get("artist")), 1.0))
                      for tracks in by_artist.values() for t in tracks
                      if track_identity(t) not in recent_ids]
        while candidates and len(result.tracks) < result.requested_count:
            pick = _weighted_pick(rng, candidates)
            if pick is None:
                break
            result.tracks.append(pick)
            chosen_ids.add(track_identity(pick))
            picked_per_artist[str(pick.get("artist") or "")] = \
                picked_per_artist.get(str(pick.get("artist") or ""), 0) + 1
            candidates = [(t, w) for t, w in candidates
                          if track_identity(t) not in chosen_ids
                          and not _capped(str(t.get("artist") or ""))]
        if result.tracks:
            result.relaxation_pass = CONTINUITY_FLOOR
            if CONTINUITY_FLOOR not in result.passes_used:
                result.passes_used.append(CONTINUITY_FLOOR)

    result.returned_count = len(result.tracks)
    result.recent_excluded_count = len(recent_excluded)
    result.generated_excluded_count = len(generated_excluded)
    result.exhausted = result.returned_count < result.requested_count
    return result
