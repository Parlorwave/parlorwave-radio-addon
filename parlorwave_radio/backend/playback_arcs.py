#!/usr/bin/env python3
"""Session-arc logic for Playback Intelligence.

This module owns everything about *arcs* -- template definitions, stage
transitions, target-energy math, observed-energy adaptation, scorer-weight
overlays, mode keyword fallback and closed-vocabulary validation -- as pure
data and stateless functions.

Boundary contract (binding):
- No I/O of any kind: no sqlite3, no musicdb, no urllib, no filesystem.
  `playback.py` reads state, calls in here to compute, and applies the result.
  A purity test (tests/test_playback_arcs.py) enforces the import graph.
- `playback.py` -> `playback_arcs.py` is the only dependency direction.
- The template keys below are the closed mode vocabulary (Principle 5): an LLM
  may *select* one of these, never invent one.
"""

from typing import Any

# Scoring-weight defaults (moved verbatim from playback.py; playback.py
# re-exports it so existing references keep working).
# Arc-stage weight profiles overlay these.
DEFAULT_WEIGHTS = {
    "request_fit": 3, "artist_cooldown": 3, "album_cooldown": 2, "track_cooldown": 3,
    "genre_continuity": 1, "genre_exploration": 1,
    "decade_continuity": 1, "decade_exploration": 1,
    "energy_fit": 2, "affinity": 2, "familiarity": 1, "live_demo": 2,
    # forgotten_bonus went 0 -> 1 when the forgotten-favorites lane arrived: a MODEST influence
    # everywhere (the component only ever adds, [0.5, 1.0]); rediscover weights
    # it 4 and the 1-in-6 blend cap bounds it in every other template.
    "skip_recovery": 2, "session_diversity": 2, "forgotten_bonus": 1, "arc_fit": 2,
}

# Observed energy = mean of the last N session tracks with outcome='played' AND a
# known energy (skipped/partial/pending/queued NEVER count -- played-only, binding).
# N=5: >= the 3-track adaptation streak, small enough to follow the listener
# within a stage.
OBSERVED_ENERGY_WINDOW = 5

# Projected artist-share cap (played + intended queue), template/stage-overridable.
DEFAULT_ARTIST_SHARE_CAP = 0.20

# Adaptation rule: if the last ADAPT_STREAK
# qualifying played tracks each deviate from target by > ADAPT_DEVIATION, shift
# the target ADAPT_SHIFT toward observed.
ADAPT_DEVIATION = 0.2
ADAPT_STREAK = 3
ADAPT_SHIFT = 0.1


# --- Templates (the closed mode vocabulary) -----------------------------------
#
# Stage schema: {key, target_energy: (lo, hi) | None (follow observed),
#   familiarity_bias 0..1, exploration_rate 0..1, weight_overrides {...},
#   advance_after {tracks: N | minutes: M}?, advance_when: predicate-key?,
#   next: stage-key? (loops), hard_caps {max_energy?, exclude_live_demo?}?,
#   artist_share_cap? }.
# familiarity_bias/exploration_rate translate to weights (see weights_for):
#   familiarity weight = 2*bias; genre/decade exploration weight = 2*rate --
#   0.5/0.5 reproduces DEFAULT_WEIGHTS exactly. Explicit weight_overrides win.

TEMPLATES: dict[str, dict[str, Any]] = {
    "default_adaptive": {
        "display_name": "Default Adaptive",
        "stages": [
            {"key": "adaptive", "target_energy": None,   # target follows observed
             "familiarity_bias": 0.5, "exploration_rate": 0.5,
             "weight_overrides": {}},
        ],
    },
    "morning_coffee": {
        "display_name": "Morning Coffee",
        # High familiarity, near-zero exploration, no live/demo.
        "stages": [
            {"key": "wake", "target_energy": (0.20, 0.35),
             "familiarity_bias": 0.8, "exploration_rate": 0.05,
             "hard_caps": {"exclude_live_demo": True},
             "advance_after": {"tracks": 3}},
            {"key": "warm", "target_energy": (0.30, 0.50),
             "familiarity_bias": 0.8, "exploration_rate": 0.05,
             "hard_caps": {"exclude_live_demo": True},
             "advance_after": {"tracks": 5}},
            {"key": "day", "target_energy": (0.40, 0.60),
             "familiarity_bias": 0.8, "exploration_rate": 0.05,
             "hard_caps": {"exclude_live_demo": True}},          # open-ended
        ],
    },
    "cleaning": {
        "display_name": "Cleaning",
        # Energetic; artist-diversity weight x1.5.
        "stages": [
            {"key": "ramp", "target_energy": (0.50, 0.70),
             "familiarity_bias": 0.6, "exploration_rate": 0.3,
             "weight_overrides": {"session_diversity": 3},
             "advance_after": {"tracks": 3}},
            {"key": "drive", "target_energy": (0.70, 0.90),
             "familiarity_bias": 0.6, "exploration_rate": 0.3,
             "weight_overrides": {"session_diversity": 3}},      # open-ended
        ],
    },
    "road_trip": {
        "display_name": "Road Trip",
        # Long-session template; loops cruise <-> singalong.
        "stages": [
            {"key": "rollout", "target_energy": (0.50, 0.70),
             "familiarity_bias": 0.6, "exploration_rate": 0.3,
             "advance_after": {"tracks": 4}},
            {"key": "cruise", "target_energy": (0.60, 0.80),
             "familiarity_bias": 0.6, "exploration_rate": 0.3,
             "advance_after": {"minutes": 40}},
            {"key": "singalong", "target_energy": (0.70, 0.90),
             "familiarity_bias": 0.9, "exploration_rate": 0.1,
             "weight_overrides": {"affinity": 3},                # thumbed anthems
             "advance_after": {"tracks": 6}, "next": "cruise"},  # loop back
        ],
    },
    "dinner": {
        "display_name": "Dinner",
        # Strict energy ceiling at the table (hard cap); winds down
        # after 21:30 when the session opted into time context.
        "stages": [
            {"key": "settle", "target_energy": (0.30, 0.45),
             "familiarity_bias": 0.6, "exploration_rate": 0.1,
             "weight_overrides": {"energy_fit": 3, "genre_continuity": 2},
             "advance_after": {"tracks": 3}},
            {"key": "table", "target_energy": (0.25, 0.45),
             "familiarity_bias": 0.6, "exploration_rate": 0.1,
             "weight_overrides": {"energy_fit": 3, "genre_continuity": 2},
             "hard_caps": {"max_energy": 0.6},
             "advance_when": "after_2130"},
            {"key": "wind_down", "target_energy": (0.20, 0.35),
             "familiarity_bias": 0.7, "exploration_rate": 0.05,
             "weight_overrides": {"energy_fit": 3},
             "hard_caps": {"max_energy": 0.5}},                  # terminal
        ],
    },
    "party": {
        "display_name": "Party",
        # Parties want anthems; skips are urgent signals (skip_recovery x2);
        # the peak loosens the projected artist-share cap.
        "stages": [
            {"key": "arrivals", "target_energy": (0.50, 0.70),
             "familiarity_bias": 0.7, "exploration_rate": 0.2,
             "weight_overrides": {"skip_recovery": 4},
             "advance_after": {"minutes": 30}},
            {"key": "peak", "target_energy": (0.75, 0.95),
             "familiarity_bias": 0.85, "exploration_rate": 0.1,
             "weight_overrides": {"skip_recovery": 4},
             "artist_share_cap": 0.30,
             "advance_after": {"minutes": 120}},
            {"key": "late", "target_energy": (0.60, 0.80),
             "familiarity_bias": 0.8, "exploration_rate": 0.1,
             "weight_overrides": {"skip_recovery": 4}},          # open-ended
        ],
    },
    "rediscover": {
        "display_name": "Rediscover My Music",
        # The Forgotten Favorites showcase: the lane's bonus
        # dominates and the blend cap is lifted (every slot may be a rediscovery).
        "forgotten_blend_per": 1,
        "stages": [
            {"key": "rediscover", "target_energy": None,         # follow listener
             "familiarity_bias": 0.5, "exploration_rate": 0.2,
             "weight_overrides": {"forgotten_bonus": 4}},
        ],
    },
    "mellow": {
        "display_name": "Mellow",
        # Hard energy ceiling; live/demo excluded; continuity x2.
        "stages": [
            {"key": "mellow", "target_energy": (0.15, 0.40),
             "familiarity_bias": 0.6, "exploration_rate": 0.2,
             "weight_overrides": {"genre_continuity": 2, "decade_continuity": 2},
             "hard_caps": {"max_energy": 0.5, "exclude_live_demo": True}},
        ],
    },
}

VALID_MODES = frozenset(TEMPLATES)


def template_for(mode: str | None) -> dict[str, Any]:
    """The template for a mode; unknown/legacy mode strings run Default Adaptive
    (the stored mode string is preserved by the caller -- only the arc behavior
    falls back)."""
    return TEMPLATES.get(mode or "") or TEMPLATES["default_adaptive"]


def initial_stage(mode: str | None) -> dict[str, Any]:
    return template_for(mode)["stages"][0]


def stage_for(mode: str | None, stage_key: str | None) -> dict[str, Any]:
    """The stage dict for (mode, key); falls back to the template's first stage
    when the key is unknown (e.g. a schema-archived session resumed)."""
    template = template_for(mode)
    for stage in template["stages"]:
        if stage["key"] == stage_key:
            return stage
    return template["stages"][0]


def next_stage_key(mode: str | None, stage_key: str | None) -> str | None:
    """The stage after `stage_key`: an explicit `next` pointer wins (loops, e.g.
    Road Trip cruise<->singalong); otherwise the following list entry; None for a
    terminal stage (stay)."""
    stages = template_for(mode)["stages"]
    for i, stage in enumerate(stages):
        if stage["key"] == stage_key:
            if stage.get("next"):
                return stage["next"]
            return stages[i + 1]["key"] if i + 1 < len(stages) else None
    return None


def _predicate(key: str, now: Any, context: dict[str, Any] | None) -> bool:
    """advance_when predicates -- computable only from what the engine has
    (clock, context snapshot). Unknown predicate keys never fire."""
    if key == "after_2130":
        # Fires only when the session opted into time context and the
        # wall clock (caller-passed, testable) is at/after 21:30.
        if not (context or {}).get("time_of_day"):
            return False
        return now is not None and (now.hour, now.minute) >= (21, 30)
    return False


def should_advance(stage: dict[str, Any], tracks_in_stage: int,
                   minutes_in_stage: float, now: Any = None,
                   context: dict[str, Any] | None = None) -> str | None:
    """Return the transition trigger (a grounded string for the session event)
    or None to stay. Order: track count, elapsed minutes, predicate."""
    advance = stage.get("advance_after") or {}
    n = advance.get("tracks")
    if n is not None and tracks_in_stage >= n:
        return f"tracks>={n}"
    m = advance.get("minutes")
    if m is not None and minutes_in_stage >= m:
        return f"minutes>={m}"
    predicate = stage.get("advance_when")
    if predicate and _predicate(predicate, now, context):
        return predicate
    return None


def target_energy_for(stage: dict[str, Any], observed: float | None) -> float | None:
    """A stage's target energy: band midpoint; a band of None means 'follow the
    listener' (Default Adaptive) -- target = observed, None until a play exists."""
    band = stage.get("target_energy")
    if band is None:
        return None if observed is None else round(max(0.0, min(1.0, float(observed))), 4)
    lo, hi = band
    return round((lo + hi) / 2.0, 4)


def weights_for(mode: str | None, stage_key: str | None) -> dict[str, float]:
    """DEFAULT_WEIGHTS + the stage's familiarity/exploration translation + its
    explicit weight_overrides (which win). Pure; the scorer applies the result."""
    stage = stage_for(mode, stage_key)
    w = dict(DEFAULT_WEIGHTS)
    bias = stage.get("familiarity_bias")
    if bias is not None:
        w["familiarity"] = round(2.0 * bias, 2)
    rate = stage.get("exploration_rate")
    if rate is not None:
        w["genre_exploration"] = round(2.0 * rate, 2)
        w["decade_exploration"] = round(2.0 * rate, 2)
    w.update(stage.get("weight_overrides") or {})
    return w


def hard_caps_for(mode: str | None, stage_key: str | None) -> dict[str, Any]:
    """Stage hard caps ({} when none): retrieval-time exclusions, never score-0."""
    return dict(stage_for(mode, stage_key).get("hard_caps") or {})


def artist_share_cap_for(mode: str | None, stage_key: str | None = None) -> float:
    """Projected played+queued artist-share cap: stage override > template
    override > default 0.20."""
    template = template_for(mode)
    stage = stage_for(mode, stage_key)
    return float(stage.get("artist_share_cap",
                           template.get("artist_share_cap", DEFAULT_ARTIST_SHARE_CAP)))


def adapt_target(target: float | None,
                 recent_played_energies: list[float]) -> float | None:
    """Observed-energy adaptation: if the last ADAPT_STREAK qualifying
    played tracks each deviate from the current target by > ADAPT_DEVIATION,
    all on the same side, shift the target ADAPT_SHIFT toward them (follow the
    listener, don't force the template). Returns the new target or None for no
    shift. Qualifying = resolved 'played' with a known energy; the caller
    supplies exactly those. Pure."""
    if target is None or len(recent_played_energies) < ADAPT_STREAK:
        return None
    tail = recent_played_energies[-ADAPT_STREAK:]
    deltas = [e - target for e in tail]
    if all(d > ADAPT_DEVIATION for d in deltas):
        return round(min(1.0, target + ADAPT_SHIFT), 4)
    if all(d < -ADAPT_DEVIATION for d in deltas):
        return round(max(0.0, target - ADAPT_SHIFT), 4)
    return None


def projected_share(artist_played: int, artist_queued: int,
                    total_played: int, total_queued: int) -> float:
    """Projected artist share across played + intended queue, INCLUDING the
    candidate under consideration (+1 both sides): a large pre-built queue can
    never hide an artist's overrepresentation behind not-yet-played rows."""
    return (artist_played + artist_queued + 1) / (total_played + total_queued + 1)


def diversity_value(share: float, cap: float) -> float:
    """session_diversity component from a projected share vs the cap: 1.0 at or
    under the cap, sloping to 0 beyond it (the same slope the retired scorer used)."""
    if share <= cap:
        return 1.0
    return max(0.0, 1.0 - (share - cap) * 2.5)


def forgotten_blend_for(mode: str | None) -> int:
    """Forgotten-pick blend cap for a template: at most 1 forgotten pick per N
    batch slots (default 6); <= 1 means
    unlimited (rediscover)."""
    return int(template_for(mode).get("forgotten_blend_per", 6))


# --- Mode selection (session start only) --------------------------------------
#
# Priority: explicit API mode -> optional bounded LLM mapping -> keyword fallback
# -> default_adaptive. The LLM (injected as llm_fn; this module stays I/O-free)
# is an opinion mapped onto the closed vocabulary: anything invalid, empty, or
# raising falls through SILENTLY (Principle 6). Never called by /next, /refill,
# /event or GET paths -- they never construct an llm_fn.

# Ordered keyword table over normalized utterance text (first match wins;
# longer/more specific phrases before generic single words).
KEYWORD_MODES: list[tuple[str, str]] = [
    ("road trip", "road_trip"), ("roadtrip", "road_trip"), ("driving", "road_trip"),
    ("drive", "road_trip"), ("highway", "road_trip"),
    ("rediscover", "rediscover"), ("deep cuts", "rediscover"),
    ("forgotten", "rediscover"), ("forgot", "rediscover"),
    ("cleaning", "cleaning"), ("chores", "cleaning"), ("housework", "cleaning"),
    ("clean", "cleaning"),
    ("dinner", "dinner"), ("supper", "dinner"), ("cooking", "dinner"),
    ("party", "party"), ("celebration", "party"),
    ("wind down", "mellow"), ("unwind", "mellow"), ("mellow", "mellow"),
    ("chill", "mellow"), ("relax", "mellow"), ("calm", "mellow"), ("quiet", "mellow"),
    ("morning", "morning_coffee"), ("coffee", "morning_coffee"),
    ("breakfast", "morning_coffee"), ("wake up", "morning_coffee"),
]


def _norm_utterance(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in (text or "").lower())
                    .split())


def keyword_mode(utterance: str) -> str | None:
    """First keyword-table hit in the normalized utterance, or None."""
    text = f" {_norm_utterance(utterance)} "
    if not text.strip():
        return None
    for phrase, mode in KEYWORD_MODES:
        if f" {phrase} " in text or (len(phrase) >= 5 and phrase in text):
            return mode
    return None


def resolve_mode(explicit: str | None, utterance: str = "",
                 llm_fn: Any = None) -> tuple[str, str]:
    """Resolve the session mode. Returns (mode, source) where source is one of
    'explicit' | 'llm' | 'keyword' | 'default'. Every fallback is silent."""
    if explicit:
        candidate = explicit.strip().lower()
        if candidate in VALID_MODES:
            return candidate, "explicit"
    if llm_fn is not None:
        try:
            raw = llm_fn(utterance)
        except Exception:
            raw = None
        if raw:
            candidate = str(raw).strip().lower()
            if candidate in VALID_MODES:   # closed-vocabulary validation
                return candidate, "llm"
    hit = keyword_mode(utterance)
    if hit:
        return hit, "keyword"
    return "default_adaptive", "default"
