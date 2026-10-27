"""Station feel: the listener's programming preference, as refill POLICY.

Pure module. A "feel" names a RefillPolicy for station_selector -- the
documented runtime seam that shapes how an active station's frozen artist
roster is played out (rotation windows, spacing, score tier, how hard the
roster's weights are followed) without touching the roster itself. Station
identity (artist_weights, seed, request text) is never read or written
here; a feel change alters only what the NEXT replenishment picks.

What the current backend can honestly express is the familiarity axis,
grounded in PERSONAL familiarity measured from play history
(station_features: how often this household actually heard each song
through), never in public chart data, which no local source provides:

  familiar   Familiar favorites. Stay close to the station's core: the
             strongest artists in the roster dominate more, songs the
             household has heard through pull extra weight
             (familiarity_power), songs the station already generated are
             eligible again sooner, and the rotation memory is shorter, so
             the same favorites come around more often.
  balanced   Today's behaviour, exactly (DEFAULT_POLICY): familiarity is
             computed but NOT weighted, and no anchor cadence applies.
  discovery  Discovery. Spread across the whole roster: weights are
             flattened so lower-weight artists get real airtime, previously
             generated songs stay excluded longer, artists and albums are
             spaced further apart, and no score-tier cut is applied, so an
             artist's deeper catalog gets the same chance as its best-rated
             tracks. Rotation memory is longer, so repeats are rarer. The
             safety line: no run of ANCHOR_EVERY consecutive picks may be
             all-unrecognisable -- when one threatens, the next pick is
             restricted to tracks the household knows (familiarity >=
             station_features.ANCHOR_MIN_FAMILIARITY) if any candidate
             qualifies, so exploration keeps returning to recognisable
             ground instead of drifting into an hour of strangers.

Energy-based feels (mellow / upbeat / high energy) are deliberately NOT
offered: the catalog carries no energy per track and the station selector
has no energy input, so such a preset would be a label with no behaviour
behind it. The retired track scorer is not the answer either. When a real
energy signal exists, it lands as further policy values here.

Effects that are only statistical (a weighting, a shorter window) are still
deterministic for a given RNG seed and pinned that way by tests.
"""

from __future__ import annotations

from typing import Any

from station_features import ANCHOR_MIN_FAMILIARITY
from station_selector import (BROADENED, CONTINUITY_FLOOR, DEFAULT_POLICY,
                              PREFERRED, SHORT_ROTATION, Pass, RefillPolicy)

# Discovery's anchor cadence: at least one recognisable track in every
# window of this many picks (when the roster can provide one at all).
ANCHOR_EVERY = 4

FEELS = ("familiar", "balanced", "discovery")
DEFAULT_FEEL = "balanced"

_CATALOG = (
    {"id": "familiar", "name": "Familiar favorites",
     "description": ("Stick close to the station's core artists and best "
                     "material; favorites come around more often.")},
    {"id": "balanced", "name": "Balanced",
     "description": "The standard rotation: spaced out, no strong lean."},
    {"id": "discovery", "name": "Discovery",
     "description": ("Spread across the whole roster and deeper into each "
                     "artist's catalog; fewer repeats.")},
)

_FLOOR = DEFAULT_POLICY.passes[3]

FAMILIAR_POLICY = RefillPolicy(passes=(
    # Shorter rotation memory, the generated ring merely discouraged rather
    # than excluded, the roster's weights followed harder (power 1.5: a 10:2
    # profile leans ~11:1 before spacing instead of 5:1), and songs the
    # household has heard through pull extra weight (familiarity_power 1.0:
    # a fully familiar track outweighs an unheard one 5:1, (0.25+1)/(0.25+0),
    # inside the same artist/spacing rules -- never a ban on the unheard).
    Pass(PREFERRED, 2.0, 20, False, 0.6, 0.35, 3, 0.25, 4, 0.6, 1.5, 1.0),
    Pass(BROADENED, 2.0, 20, False, 0.8, 0.50, 3, 1.0, 1, 1.0, 1.5, 1.0),
    Pass(SHORT_ROTATION, 1.0, 10, False, 1.0, 0.70, 2, 1.0, 1, 1.0, 1.0),
    _FLOOR,
))

DISCOVERY_POLICY = RefillPolicy(anchor_every=ANCHOR_EVERY,
                                anchor_min=ANCHOR_MIN_FAMILIARITY, passes=(
    # Longer memory, generated ring excluded through two rungs, stricter
    # artist/album spacing, no tier cut, and flattened weights (power 0.5:
    # a 10:2 profile leans ~2.2:1 before spacing instead of 5:1).
    Pass(PREFERRED, 8.0, 80, True, 1.0, 0.25, 4, 0.20, 5, 1.0, 0.5),
    Pass(BROADENED, 6.0, 60, True, 1.0, 0.40, 3, 1.0, 1, 1.0, 0.5),
    Pass(SHORT_ROTATION, 2.0, 20, False, 0.20, 0.60, 2, 1.0, 1, 1.0, 0.75),
    _FLOOR,
))

POLICIES: dict[str, RefillPolicy] = {
    "familiar": FAMILIAR_POLICY,
    "balanced": DEFAULT_POLICY,
    "discovery": DISCOVERY_POLICY,
}


def policy_for(feel: str | None) -> RefillPolicy:
    """The policy for a feel; unknown/None falls back to balanced so an
    unreadable setting never changes what the station has always done."""
    return POLICIES.get(str(feel or ""), DEFAULT_POLICY)


def current_feel(settings_doc: dict[str, Any] | None = None) -> str:
    """The stored feel, read fresh. Any failure reads as balanced: a
    settings hiccup must never stop a refill or silently change taste."""
    try:
        doc = settings_doc
        if doc is None:
            import settings
            doc = settings.load()
        feel = (doc.get("station") or {}).get("feel")
        return feel if feel in FEELS else DEFAULT_FEEL
    except Exception:
        return DEFAULT_FEEL


def catalog() -> list[dict[str, Any]]:
    return [dict(item) for item in _CATALOG]
