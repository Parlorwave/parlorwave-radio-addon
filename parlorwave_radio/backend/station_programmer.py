"""The optional LLM radio programmer. REORDER ONLY.

Jarvis's deterministic programmer (eligibility -> features -> feel policy ->
selection) remains authoritative and sufficient: it alone decides WHAT is
in a refill batch, and Endless Radio never waits for a model. This module
adds one optional pass over an already-valid batch: a local LLM receives
the selected tracks with their honest features and proposes a better PLAY
ORDER -- the sequencing judgment of a radio programmer rather than the
incidental order of weighted sampling.

AUTHORITY CONTRACT (the whole point). The model may permute the selected
identities; it may do nothing else. It cannot invent, add, remove or
duplicate tracks, change the refill count, touch ownership, eligibility,
transport, manual queue items or DJ/news elements -- structurally, because
its output is mapped through a local id table and rejected unless it is a
PERFECT PERMUTATION of the input ids. "Mostly valid" is invalid; the
fallback is always the deterministic order that already exists. And
station_refill._programmer_order re-validates every identity downstream,
so even a bug HERE cannot leak an unselected track.

STRUCTURED OUTPUT. The failure mode is known (a JSON object where
an array was asked for). Ollama >= 0.5 supports schema-constrained
generation: the request carries OUTPUT_SCHEMA in `format`, so the decoder
itself cannot emit prose or a stray shape. Validation still runs in full --
the schema constrains shape, never id truthfulness. One call, no retry
loop: a refill path never queues repeated model attempts.

SCHEDULING / GPU PRIORITY. DJ and news deadlines outrank clever ordering.
The advisor runs only when dj_engine.render_pressure() says no spoken
render is in flight or imminent for this queue within the advisor's own
time budget; otherwise it reports `busy_dj` and the deterministic order
ships untouched. The production model is deliberately the SAME model as
the DJ writer: sharing weights means a refill never evicts
the writer from VRAM and a break render never pays a model reload for our
sake. The call happens in the unserialized selection phase of the refill
transaction -- no station lock, no ingestion thread, no observer is ever
held across an inference.

The model's own broad knowledge of what songs are widely recognisable MAY
inform its ordering (that is soft judgment inside a validated permutation);
it is never persisted as popularity metadata, and public_hit remains None
until a trusted source exists.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import config as _config

logger = logging.getLogger("musicbrain")

# Timeout for the ONE inference call (a warm resident model, 12 candidates,
# schema-constrained, answers well inside it). The
# refill runs off-thread at a track boundary with >= 8 playable tracks of
# runway, so seconds are affordable; minutes are not.
DEFAULT_TIMEOUT_S = 12.0
# The advisor's model: settings station.programmer.model, else the configured
# fallback (config.py). "" = no model configured = the deterministic order
# ships, reported as ST_NO_MODEL.
DEFAULT_MODEL = _config.current().programmer_model

# How much of a margin beyond our own timeout to demand from the DJ lane:
# if a spoken render would want the GPU within (timeout + this), yield now.
DJ_PRESSURE_SLACK_S = 30.0

# Bounded recent-history context: enough to avoid locally stupid
# ordering, never the listening history.
RECENT_CONTEXT = 8

ST_DISABLED = "disabled"
ST_NO_MODEL = "no_model"
ST_SUCCESS = "success"
ST_TIMEOUT = "timeout"
ST_BUSY_DJ = "busy_dj"
ST_INVALID_JSON = "invalid_json"
ST_INVALID_IDS = "invalid_ids"
ST_EMPTY = "empty"
ST_EXCEPTION = "exception"
ST_TOO_FEW = "too_few_tracks"
ST_ANCHOR_TUNNEL = "anchor_cadence_violated"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ordered_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ordered_ids"],
}

# Station-feel guidance: the deterministic station-feel semantics, spoken as
# programmer instructions. The deterministic selection already enforced the
# feel's hard behavior (weights, spacing, anchor cadence); these shape the
# SEQUENCE without permission to undermine any of it.
FEEL_GUIDANCE = {
    "balanced": (
        "Feel: balanced. Aim for coherent variety: open on something "
        "recognisable, space artists apart, mix familiar tracks with the "
        "less-known ones, keep natural momentum. Do not overthink."),
    "familiar": (
        "Feel: familiar favorites. Sequence for comfort and recognition: "
        "lean on high personal familiarity and artist affinity, and songs "
        "broadly likely to be known. Still vary artists and energy -- this "
        "is a warm rotation, not a greatest-hits countdown."),
    "discovery": (
        "Feel: discovery. Let the adventurous, low-familiarity material "
        "breathe, but obey this HARD RULE: never place more than three "
        "tracks in a row that lack the \"anchor\" flag -- at least every "
        "fourth track must be an anchor, and spacing anchors more evenly "
        "than that is better. Orders that break the rule are discarded. "
        "Spread artists widely."),
}

_PROMPT = """You are the music programmer for a home radio station. From the SAME set of already-chosen tracks, decide the best PLAY ORDER for the next stretch of the broadcast. You may only reorder; every id must appear exactly once.

{feel_guidance}

Station: {station_desc}

Recently played (oldest first, do not repeat their pattern blindly):
{recent}

Sequencing principles: open the segment well; space out same or closely related artists; avoid clumping several obscure tracks in a row; recover with something recognisable after unfamiliar stretches; end the segment cleanly. Personal familiarity (0..1, how often this household heard the song through), artist affinity (0..1) and station fit (0..1) are measured facts. "anchor" marks tracks the household reliably recognises. "rank" is the current order. You may also use your own knowledge of how widely known a song is.

Tracks to order (JSON):
{candidates}

Answer with JSON only: {{"ordered_ids": [...]}} containing every id exactly once, in play order."""


def _track_field(track: dict[str, Any], features) -> dict[str, Any]:
    entry = (features or {}).get(str(track.get("path") or ""))
    out: dict[str, Any] = {}
    for name in ("familiarity", "affinity", "fit"):
        value = getattr(entry, name, None)
        if value is not None:
            out[name] = round(float(value), 2)
    return out


def build_prompt(tracks: list[dict[str, Any]], features, feel: str,
                 *, station_desc: str = "", recent: list[str] | None = None,
                 anchor_min: float = 0.5) -> tuple[str, dict[str, str]]:
    """(prompt, id_map) where id_map maps the short candidate ids ('t01'..)
    back to track identities (catalog paths). Only honest fields are sent:
    no BPM, no energy, no mood, no fabricated popularity."""
    id_map: dict[str, str] = {}
    candidates = []
    for rank, track in enumerate(tracks, start=1):
        cid = f"t{rank:02d}"
        id_map[cid] = str(track.get("path") or "")
        row: dict[str, Any] = {"id": cid, "rank": rank,
                               "artist": track.get("artist") or "",
                               "title": track.get("title") or ""}
        if track.get("album"):
            row["album"] = track["album"]
        if track.get("year"):
            row["year"] = track["year"]
        row.update(_track_field(track, features))
        if row.get("familiarity", 0) >= anchor_min:
            row["anchor"] = True
        candidates.append(row)
    recent_lines = "\n".join(f"- {r}" for r in (recent or [])[-RECENT_CONTEXT:])
    prompt = _PROMPT.format(
        feel_guidance=FEEL_GUIDANCE.get(feel, FEEL_GUIDANCE["balanced"]),
        station_desc=station_desc or "a personal music station",
        recent=recent_lines or "- (nothing recorded yet)",
        candidates=json.dumps(candidates, ensure_ascii=False))
    return prompt, id_map


def validate(raw: Any, id_map: dict[str, str]) -> tuple[list[str] | None, str]:
    """(ordered track identities, '') or (None, status). A PERFECT
    permutation of the candidate ids or nothing: every expected id exactly
    once, no unknown ids, no duplicates, no omissions."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None, ST_INVALID_JSON
    if isinstance(raw, dict):
        raw = raw.get("ordered_ids")
    if raw is None:
        return None, ST_INVALID_JSON
    if not isinstance(raw, list):
        return None, ST_INVALID_JSON
    if not raw:
        return None, ST_EMPTY
    ids = [str(x) for x in raw]
    if len(ids) != len(id_map) or set(ids) != set(id_map):
        return None, ST_INVALID_IDS
    return [id_map[i] for i in ids], ""


def _anchor_cadence_ok(ordered: list[str], features, feel: str,
                       anchor_min: float = 0.5, run_limit: int = 4) -> bool:
    """Discovery's anchor principle applies to the ORDER too: the
    deterministic selection guaranteed no long run of unrecognisable picks,
    and a permutation must not rebuild that tunnel. Only enforced when the
    batch actually contains an anchor to place -- an all-unfamiliar batch
    (a young household) cannot honour a cadence and is not rejected for
    it."""
    if feel != "discovery":
        return True

    def fam(path: str) -> float:
        value = getattr((features or {}).get(path), "familiarity", None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if not any(fam(p) >= anchor_min for p in ordered):
        return True
    run = 0
    for path in ordered:
        run = 0 if fam(path) >= anchor_min else run + 1
        if run >= run_limit:
            return False
    return True


def _settings() -> dict[str, Any]:
    try:
        import settings
        cfg = settings.load()["station"].get("programmer") or {}
    except Exception:
        return {"enabled": False}
    return cfg


def reorder(tracks: list[dict[str, Any]], features, feel: str, *,
            station_desc: str = "", recent: list[str] | None = None,
            queue_id: str = "", now=None, db_path=None,
            generate_fn=None, pressure_fn=None,
            config: dict[str, Any] | None = None) -> dict[str, Any]:
    """One optional advisory pass. Returns
    {"ordered": [track identities] | None, "telemetry": {...}} -- `ordered`
    is None whenever the deterministic order should stand, and the
    telemetry says exactly why. NEVER raises."""
    telemetry: dict[str, Any] = {"used": False, "status": ST_DISABLED,
                                 "model": None, "latency_ms": None,
                                 "candidate_count": 0,
                                 "feel": feel, "order_after": None}
    try:
        telemetry["candidate_count"] = len(tracks)
        cfg = config if config is not None else _settings()
        if not cfg.get("enabled"):
            return {"ordered": None, "telemetry": telemetry}
        model = str(cfg.get("model") or DEFAULT_MODEL)
        timeout = float(cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
        telemetry["model"] = model
        if generate_fn is None and not model:
            telemetry["status"] = ST_NO_MODEL
            return {"ordered": None, "telemetry": telemetry}
        if len(tracks) < 3:
            telemetry["status"] = ST_TOO_FEW
            return {"ordered": None, "telemetry": telemetry}

        # GPU priority: a spoken render in flight or due within
        # our own budget owns the model. Yield instantly, no queueing.
        try:
            if pressure_fn is None:
                import dj_engine
                pressure = dj_engine.render_pressure(
                    queue_id, now=now,
                    horizon_s=timeout + DJ_PRESSURE_SLACK_S, db_path=db_path)
            else:
                pressure = pressure_fn(timeout + DJ_PRESSURE_SLACK_S)
        except Exception:
            pressure = False        # an unreadable DJ lane never blocks music
        if pressure:
            telemetry["status"] = ST_BUSY_DJ
            return {"ordered": None, "telemetry": telemetry}

        prompt, id_map = build_prompt(tracks, features, feel,
                                      station_desc=station_desc,
                                      recent=recent)
        started = time.monotonic()
        try:
            if generate_fn is None:
                import providers
                body = providers.ollama_generate(prompt, model,
                                                 timeout=timeout,
                                                 format=OUTPUT_SCHEMA)
            else:
                body = generate_fn(prompt, model, timeout=timeout,
                                   format=OUTPUT_SCHEMA)
        except TimeoutError:
            telemetry["status"] = ST_TIMEOUT
            return {"ordered": None, "telemetry": telemetry}
        except Exception as exc:  # noqa: BLE001 - one inference boundary
            telemetry["status"] = (ST_TIMEOUT if "timed out" in str(exc)
                                   else ST_EXCEPTION)
            return {"ordered": None, "telemetry": telemetry}
        finally:
            telemetry["latency_ms"] = int(
                (time.monotonic() - started) * 1000)

        ordered, status = validate((body or {}).get("response"), id_map)
        if ordered is not None and not _anchor_cadence_ok(ordered, features,
                                                          feel):
            ordered, status = None, ST_ANCHOR_TUNNEL
        if ordered is None:
            telemetry["status"] = status
            logger.info("station programmer fallback: %s (model=%s, %sms)",
                        status, model, telemetry["latency_ms"])
            return {"ordered": None, "telemetry": telemetry}
        telemetry["used"] = True
        telemetry["status"] = ST_SUCCESS
        # The permutation in candidate-id space ('t04 t01 ...'): the full
        # before/after story in a few bytes, since tNN IS deterministic
        # rank NN. No prose, no reasoning, nothing model-authored persists.
        by_path = {v: k for k, v in id_map.items()}
        telemetry["order_after"] = [by_path[p] for p in ordered]
        return {"ordered": ordered, "telemetry": telemetry}
    except Exception:  # noqa: BLE001 - the advisor must NEVER take down refill
        logger.warning("station programmer failed (deterministic order "
                       "keeps)", exc_info=True)
        telemetry["status"] = ST_EXCEPTION
        return {"ordered": None, "telemetry": telemetry}
