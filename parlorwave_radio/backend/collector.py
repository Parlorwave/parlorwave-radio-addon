#!/usr/bin/env python3
"""Collector Intelligence analyzer (Phase 1: grounded completion & gaps).

Compares the owned library (music_index.json) against each artist's MusicBrainz
studio discography and materializes per-artist completion facts into a derived,
disposable collection.db. No estimates, no LLM, no Lidarr. MusicBrainz is the
sole source of truth for which albums exist; an album is only ever "missing" if
MB lists a studio release group that is not matched locally.

Usage:
    python -m collector build [--index PATH] [--db PATH] [--limit N]
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

from musicbrainz_client import lookup_artist_release_groups
from release_filters import is_studio_album, match_key, normalize_title

from paths import BASE_DIR

BASE = BASE_DIR
INDEX = BASE / "music_index.json"
DB = BASE / "collection.db"

# A --stale-only build refreshes artists missing from ci_artists or whose row is
# older than this window (days). Aligned with the summary freshness threshold so
# a nightly stale-only build keeps the collection reported-fresh; network cost is
# bounded by the 30-day MusicBrainz cache TTL, which absorbs most refreshes.
REFRESH_DAYS = float(os.environ.get("COLLECTOR_REFRESH_DAYS", "7"))

# The collection is reported stale once its last build is older than this many
# days (documented freshness threshold; the summary surfaces stale + age_days).
STALE_AFTER_DAYS = float(os.environ.get("COLLECTOR_STALE_AFTER_DAYS", "7"))

# Recency half-life (days) for the affinity play signal: a play decays to half
# weight after this many days. Only last_played carries a reliable timestamp.
AFFINITY_HALFLIFE_DAYS = float(os.environ.get("AFFINITY_HALFLIFE_DAYS", "180"))


def _recency_weight(last_played: str | None, as_of: datetime) -> float:
    """0..1 decayed play-recency from last_played (0.0 if never played)."""
    if not last_played:
        return 0.0
    try:
        played_at = datetime.fromisoformat(last_played)
    except ValueError:
        return 0.0
    age_days = max(0.0, (as_of - played_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / AFFINITY_HALFLIFE_DAYS)


def affinity_score(thumbs_up: int, thumbs_down: int, skips: int,
                   recency_weight: float) -> float:
    """Bounded 0..1 affinity (0.5 neutral) from grounded local behavior. Thumbs and
    skips are timeless counts; the play signal is recency-decayed."""
    raw = 2.0 * thumbs_up - 3.0 * thumbs_down - 1.0 * skips + 2.0 * recency_weight
    return 1.0 / (1.0 + math.exp(-raw / 4.0))


def compute_affinity(inputs: dict[str, dict[str, Any]], as_of: datetime,
                     built_at: str) -> list[dict[str, Any]]:
    """Turn per-artist raw signals into affinity rows (deterministic given as_of).

    All inputs are the user's own recorded behavior -> provenance 'grounded_local'.
    """
    rows = []
    for name, s in inputs.items():
        rw = _recency_weight(s.get("last_played"), as_of)
        rows.append({
            "name": name,
            "artist_mbid": s.get("artist_mbid"),
            "thumbs_up": int(s.get("thumbs_up", 0)),
            "thumbs_down": int(s.get("thumbs_down", 0)),
            "skips": int(s.get("skips", 0)),
            "played_tracks": int(s.get("played_tracks", 0)),
            "last_played": s.get("last_played"),
            "recency_weight": round(rw, 6),
            "affinity_score": round(affinity_score(
                int(s.get("thumbs_up", 0)), int(s.get("thumbs_down", 0)),
                int(s.get("skips", 0)), rw), 6),
            "provenance": "grounded_local",
            "built_at": built_at,
        })
    return rows


def write_affinity(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    """Replace ci_affinity with the given rows (derived, regenerated each build)."""
    cur = conn.cursor()
    cur.execute("DELETE FROM ci_affinity")
    cur.executemany(
        "INSERT OR REPLACE INTO ci_affinity "
        "(name, artist_mbid, thumbs_up, thumbs_down, skips, played_tracks, "
        "last_played, recency_weight, affinity_score, provenance, built_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r["name"], r["artist_mbid"], r["thumbs_up"], r["thumbs_down"], r["skips"],
          r["played_tracks"], r["last_played"], r["recency_weight"],
          r["affinity_score"], r["provenance"], r["built_at"]) for r in rows],
    )


# --- Recommendation ledger ---------------------------------------------------
# Score rows live here (collection.db, disposable); lifecycle status lives in the
# persistent store (musicdb.recommendation_lifecycle). Both key on rec_identity so
# a rebuild reattaches user intent to regenerated score rows.

def rec_identity(lane: str, release_group_mbid: str) -> str:
    """Stable recommendation id from (lane, release_group_mbid). The single rule
    both stores use, so score rows and lifecycle rows join across rebuilds."""
    if not release_group_mbid:
        raise ValueError("recommendation identity requires a release_group_mbid")
    return f"{lane}:{release_group_mbid}"


def write_recommendations(conn: sqlite3.Connection, rows: list[dict[str, Any]],
                          run_id: str, generated_at: str) -> list[str]:
    """Insert/replace derived recommendation score rows. Each row supplies lane,
    release_group_mbid and score fields; rec_id is derived. Returns the rec_ids.
    Idempotent per (lane, release_group_mbid). Does no MB/LLM/Lidarr work."""
    cur = conn.cursor()
    rec_ids = []
    for r in rows:
        rec_id = rec_identity(r["lane"], r["release_group_mbid"])
        rec_ids.append(rec_id)
        cur.execute(
            "INSERT OR REPLACE INTO recommendations "
            "(rec_id, lane, release_group_mbid, artist_mbid, artist_name, album, "
            "year, rank, score, tier, components, provenance, score_reason, "
            "run_id, generated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rec_id, r["lane"], r["release_group_mbid"], r.get("artist_mbid"),
             r.get("artist_name"), r.get("album"), r.get("year"), r.get("rank"),
             r.get("score"), r.get("tier"),
             json.dumps(r.get("components") or {}),
             json.dumps(r.get("provenance") or {}),
             r.get("score_reason"), run_id, generated_at),
        )
    return rec_ids


def _rec_row(row: tuple, cols: list[str]) -> dict[str, Any]:
    d = dict(zip(cols, row))
    for key in ("components", "provenance"):
        try:
            d[key] = json.loads(d.get(key) or "{}")
        except (TypeError, ValueError):
            d[key] = {}
    return d


def _tier_from_score(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def completion_lane_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build ranked completion-lane recommendation rows from grounded CI data only.

    Candidates are exactly the ci_missing release groups (MB-sourced) -- no LLM, no
    MusicBrainz, no new candidates. Scored by completion opportunity, collection
    investment, and per-artist affinity. Returns (rows, counters).
    """
    artists = {
        r[0]: {"completion_pct": r[1], "owned_studio": r[2],
               "total_studio": r[3], "missing_count": r[4]}
        for r in conn.execute(
            "SELECT name, completion_pct, owned_studio, total_studio, missing_count "
            "FROM ci_artists")
    }
    affinity = {r[0]: r[1] for r in conn.execute(
        "SELECT name, affinity_score FROM ci_affinity")}
    missing = conn.execute(
        "SELECT artist_name, artist_mbid, album, year, release_group_mbid "
        "FROM ci_missing WHERE release_group_mbid IS NOT NULL").fetchall()

    rows = []
    grounded_candidates = 0
    for artist_name, artist_mbid, album, year, rg_mbid in missing:
        grounded_candidates += 1
        a = artists.get(artist_name, {})
        opportunity = (a.get("completion_pct") or 0) / 100.0
        investment = min(a.get("owned_studio") or 0, 10) / 10.0
        aff = affinity.get(artist_name)
        aff_val = aff if aff is not None else 0.5
        score = round(0.5 * opportunity + 0.2 * investment + 0.3 * aff_val, 6)
        components = {"opportunity": round(opportunity, 6),
                      "investment": round(investment, 6),
                      "affinity": round(aff_val, 6)}
        provenance = {"opportunity": "grounded", "investment": "grounded",
                      "affinity": "grounded_local" if aff is not None else "neutral"}
        reason = (f"{artist_name} {a.get('completion_pct', 0)}% complete "
                  f"({a.get('owned_studio', 0)}/{a.get('total_studio', 0)} owned, "
                  f"{a.get('missing_count', 0)} missing); investment "
                  f"{a.get('owned_studio', 0)}; affinity {aff_val:.2f}"
                  f"{'' if aff is not None else ' (neutral)'}")
        rows.append({
            "lane": "completion", "release_group_mbid": rg_mbid,
            "artist_mbid": artist_mbid, "artist_name": artist_name,
            "album": album, "year": year, "score": score,
            "tier": _tier_from_score(score), "components": components,
            "provenance": provenance, "score_reason": reason,
        })
    rows.sort(key=lambda r: (-r["score"], (r["artist_name"] or "").lower(),
                             (r["album"] or "").lower()))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    counters = {"lane": "completion", "grounded_candidates": grounded_candidates,
                "written": len(rows), "uses_llm": False}
    return rows, counters


def _replace_lane(conn: sqlite3.Connection, lane: str, rows: list[dict[str, Any]],
                  run_id: str, generated_at: str) -> None:
    """Replace all score rows for a lane with this run's rows (run semantics:
    the ledger reflects the latest run per lane). Lifecycle is never touched."""
    conn.execute("DELETE FROM recommendations WHERE lane = ?", (lane,))
    write_recommendations(conn, rows, run_id, generated_at)


def taste_fit_lane_rows(candidates: list[dict[str, Any]],
                        affinity_map: dict[str, float]
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn grounded missing-albums candidates into taste_fit ledger rows.

    The LLM only proposed artist names; every album fact here came from
    MusicBrainz grounding upstream. A candidate WITHOUT a release_group_mbid
    cannot enter the ledger (structural anti-hallucination). Adds an affinity
    ranking component (neutral for artists absent from ci_affinity).
    """
    rows = []
    grounded = 0
    dropped_ungrounded = 0
    for c in candidates:
        rg = c.get("musicbrainz_release_group_id")
        if not rg:
            dropped_ungrounded += 1
            continue  # no grounded MBID -> may not enter the ledger
        grounded += 1
        artist = c.get("artist") or ""
        album_score = int(c.get("album_score") or 0)
        album_fit = album_score / 100.0
        llm_conf = float(c.get("confidence") or 0.0)
        aff = affinity_map.get(artist)
        aff_val = aff if aff is not None else 0.5
        score = round(0.6 * album_fit + 0.3 * llm_conf + 0.1 * aff_val, 6)
        tier = ("high" if album_fit >= 0.9 and llm_conf >= 0.8
                else "medium" if score >= 0.5 else "low")
        rows.append({
            "lane": "taste_fit", "release_group_mbid": rg,
            "artist_mbid": c.get("artist_mbid"), "artist_name": artist,
            "album": c.get("album"), "year": c.get("year"), "score": score,
            "tier": tier,
            # Numeric component scores plus the display fields the frozen
            # /missing_albums shape needs (album_score, llm_reason) so serving from
            # the ledger is a clean map with no re-derivation.
            "components": {"album_fit": round(album_fit, 6),
                           "llm_confidence": round(llm_conf, 6),
                           "affinity": round(aff_val, 6),
                           "album_score": album_score,
                           "llm_reason": (c.get("reason") or "").strip()},
            "provenance": {"album_fit": "grounded", "llm_confidence": "estimated",
                           "affinity": "grounded_local" if aff is not None else "neutral"},
            # The era/type scoring explanation (the /missing_albums score_reason).
            "score_reason": (c.get("score_reason") or "").strip(),
        })
    rows.sort(key=lambda r: (-r["score"], (r["artist_name"] or "").lower(),
                             (r["album"] or "").lower()))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    counters = {"lane": "taste_fit", "grounded_candidates": grounded,
                "dropped_ungrounded": dropped_ungrounded, "written": len(rows),
                "uses_llm": True}
    return rows, counters


def _affinity_map(db_path: Path) -> dict[str, float]:
    """Read name -> affinity_score from ci_affinity (read-only, defensive)."""
    if not Path(db_path).exists():
        return {}
    try:
        conn = _connect_ro(db_path)
        rows = conn.execute("SELECT name, affinity_score FROM ci_affinity").fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        return {}


def _default_taste_fit_fn(db_path: Path):
    """Return the taste_fit lane generator: run the proven missing_albums pipeline
    (LLM artist proposals -> MB grounding -> release_filters -> scoring), then
    re-score with affinity and write ledger rows. Raises if the LLM is unavailable
    so generate() can degrade the lane safely."""
    def taste_fit_fn(prompt: str, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import library_repo
        import missing_albums
        library = library_repo.get_library()
        resp = missing_albums.build_missing_albums_response(prompt, count, "lidarr", library)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "missing_albums pipeline failed"))
        rows, counters = taste_fit_lane_rows(resp["suggestions"], _affinity_map(db_path))
        counters["pipeline_counts"] = resp.get("pipeline_counts", {})
        return rows, counters
    return taste_fit_fn


def generate(
    db_path: Path = DB,
    lanes: tuple[str, ...] = ("completion", "taste_fit"),
    prompt: str = "essential albums for my taste",
    count: int = 50,
    run_id: str | None = None,
    generated_at: str | None = None,
    taste_fit_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the recommendation generator over the given lanes, writing the ledger.

    CLI/background only; performs no request-path work. The completion lane is
    grounded-and-deterministic; the taste_fit lane is prompt-driven (LLM) and
    degrades to zero rows if unavailable. Idempotent per lane by replace-per-lane.
    """
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    run_id = run_id or f"run-{generated_at}"
    counters: dict[str, Any] = {}
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        if "completion" in lanes:
            rows, c = completion_lane_rows(conn)
            _replace_lane(conn, "completion", rows, run_id, generated_at)
            counters["completion"] = c
            log(f"completion: {c['written']} rows from {c['grounded_candidates']} grounded candidates")
        if "taste_fit" in lanes and taste_fit_fn is not None:
            try:
                rows, c = taste_fit_fn(prompt, count)
            except Exception as exc:
                rows, c = [], {"lane": "taste_fit", "written": 0,
                               "error": "llm_unavailable", "detail": str(exc)[:200]}
                log(f"taste_fit degraded (LLM unavailable): {exc}")
            _replace_lane(conn, "taste_fit", rows, run_id, generated_at)
            counters["taste_fit"] = c
            log(f"taste_fit: {c.get('written', 0)} rows")
        conn.execute(
            "INSERT OR REPLACE INTO ci_meta (key, value) VALUES ('recommend_run', ?)",
            (json.dumps({"run_id": run_id, "generated_at": generated_at,
                         "counters": counters}),))
        conn.commit()
    finally:
        conn.close()
    return {"run_id": run_id, "generated_at": generated_at, "counters": counters}


def _read_run_meta(db_path: Path) -> dict[str, Any]:
    """Return the last recommendation run's stored counters/metadata (or {})."""
    if not Path(db_path).exists():
        return {}
    try:
        conn = _connect_ro(db_path)
        row = conn.execute(
            "SELECT value FROM ci_meta WHERE key = 'recommend_run'").fetchone()
        conn.close()
        return json.loads(row[0]) if row else {}
    except (sqlite3.OperationalError, TypeError, ValueError):
        return {}


def missing_albums_from_ledger(prompt: str, count: int, format_type: str = "lidarr",
                               db_path: Path = DB,
                               stale_after_days: float = STALE_AFTER_DAYS) -> dict[str, Any]:
    """Serve /missing_albums from the taste_fit ledger -- a pure read, no LLM, no
    MusicBrainz. Shape-frozen (additive freshness fields); pipeline_counts come
    from the generation run. Results reflect the last generation run (era-filtered
    by the request prompt), not a live crawl. Cache-cold -> ok:false."""
    import missing_albums  # for era detection only (no network)

    served = get_recommendations(db_path, lane="taste_fit")
    # Cache-cold = the ledger has no taste_fit rows at all (never generated). An
    # era filter that yields zero from a populated ledger is still "built".
    if not served.get("ok") or served.get("count", 0) == 0:
        return {"ok": False, "error": "recommendations not built", "prompt": prompt,
                "count": 0, "format": format_type, "suggestions": [], "pipeline_counts": {}}

    rows = [r for r in served["recommendations"] if r.get("status") != "dismissed"]
    window = missing_albums._detect_target_window(prompt)
    if window:
        low, high = window
        rows = [r for r in rows
                if not r.get("year") or (low - 3) <= r["year"] <= (high + 3)]
    rows.sort(key=lambda r: r.get("rank") or 1_000_000)
    rows = rows[:count]

    suggestions = []
    for r in rows:
        comp = r.get("components") or {}
        mbid = r.get("release_group_mbid")
        album_score = comp.get("album_score")
        if album_score is None:
            album_score = int(round((comp.get("album_fit") or 0) * 100))
        suggestions.append({
            "artist": r.get("artist_name"),
            "album": r.get("album"),
            "year": r.get("year"),
            "musicbrainz_release_group_id": mbid,
            "album_score": album_score,
            "score_reason": r.get("score_reason"),
            "reason": comp.get("llm_reason") or r.get("score_reason"),
            "confidence": comp.get("llm_confidence"),
            "lidarr_search": f"lidarr:{mbid}" if mbid else r.get("artist_name"),
        })

    run = _read_run_meta(db_path)
    generated_at = run.get("generated_at")
    pipeline_counts = (run.get("counters", {}).get("taste_fit", {})
                       .get("pipeline_counts", {}))
    age_days, stale = _freshness(generated_at, stale_after_days)
    return {
        "ok": True, "prompt": prompt, "count": len(suggestions),
        "format": format_type, "suggestions": suggestions,
        "pipeline_counts": pipeline_counts,
        "generated_at": generated_at, "age_days": age_days, "stale": stale,
    }


def get_recommendations(db_path: Path = DB, lane: str | None = None,
                        status_fn: Callable[[list[str]], dict[str, str]] | None = None
                        ) -> dict[str, Any]:
    """Read recommendation score rows and reattach lifecycle status (default 'new')
    from the persistent store by rec_id. Read-only; no MB/LLM work."""
    if not Path(db_path).exists():
        return {"ok": False, "error": "recommendations not built"}
    try:
        conn = _connect_ro(db_path)
    except sqlite3.OperationalError:
        return {"ok": False, "error": "recommendations not built"}
    try:
        sql = "SELECT * FROM recommendations"
        params: list[Any] = []
        if lane:
            sql += " WHERE lane = ?"
            params.append(lane)
        sql += " ORDER BY rank, score DESC"
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [_rec_row(r, cols) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return {"ok": False, "error": "recommendations not built"}  # table absent
    finally:
        conn.close()
    if status_fn is None:
        import musicdb
        status_fn = musicdb.get_recommendation_statuses
    statuses = status_fn([r["rec_id"] for r in rows])
    for r in rows:
        r["status"] = statuses.get(r["rec_id"], "new")  # reattach; default new
    return {"ok": True, "count": len(rows), "recommendations": rows}

# Lockfile guarding against overlapping build jobs (cron + manual run at once).
LOCK_PATH = BASE / "collector.lock"


class BuildLockError(RuntimeError):
    """Raised when another live collector build already holds the lock."""


def _pid_alive(pid: int) -> bool:
    """Portable "is this pid a live process" probe.

    POSIX keeps the historical signal-0 probe. On Windows os.kill(pid, 0) is
    NOT a probe -- signal 0 is CTRL_C_EVENT, which gets SENT to the pid's
    console group (interrupting this very process's console) -- so query the
    process handle instead.
    """
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the process exists (another user's).
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = ctypes.wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE  # exited pids stay openable
            return True  # handle opened but no exit code -> assume alive
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False  # holder is gone -> stale
    except PermissionError:
        return True  # exists (owned by another user) -> alive
    except OSError:
        return False
    return True


def _lock_is_active(info: dict[str, Any]) -> bool:
    """True if the lockfile's recorded process is still a live build.

    Mirrors the start script's pid+cwd discipline: the pid must be alive AND
    still running from the recorded cwd (guards against a recycled pid). A dead
    or mismatched holder means the lock is stale and may be stolen.
    """
    pid = info.get("pid")
    if not isinstance(pid, int):
        return False
    if not _pid_alive(pid):
        return False  # holder is gone -> stale
    recorded_cwd = info.get("cwd")
    try:
        actual_cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return True  # cannot verify cwd -> assume active, do not steal
    return actual_cwd == recorded_cwd


@contextmanager
def build_lock(path: Path = LOCK_PATH) -> Iterator[None]:
    """Hold an exclusive build lock for the duration of the block.

    Refuses (BuildLockError) if a live build holds it; steals a stale lock left
    by a crashed build. Released on exit only if we still own it.
    """
    if path.exists():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        if _lock_is_active(info):
            raise BuildLockError(
                f"another collector build is running (pid {info.get('pid')}); "
                "refusing to start a second one")
    path.write_text(json.dumps({"pid": os.getpid(), "cwd": os.getcwd()}),
                    encoding="utf-8")
    try:
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("pid") == os.getpid():
                path.unlink()
        except Exception:
            pass


def studio_albums(releases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return MB studio albums keyed by normalized title (first seen wins).

    Filters to canonical studio albums via release_filters.is_studio_album and
    de-duplicates release groups that share a title.
    """
    result: dict[str, dict[str, Any]] = {}
    for release in releases:
        if not is_studio_album(release):
            continue
        title = str(release.get("album") or "").strip()
        if not title:
            continue
        key = match_key(title) or normalize_title(title)
        result.setdefault(key, release)
    return result


def _release_identity(releases: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the artist identity embedded on the MB releases by the client.

    The MusicBrainz client stamps every release with the resolved artist MBID,
    canonical name, and disambiguation. Returns the first present identity, or
    all-None when releases carry none (e.g. no MB data, or injected test rows).
    """
    for release in releases:
        if release.get("artist_mbid"):
            return {
                "artist_mbid": release.get("artist_mbid"),
                "mb_name": release.get("mb_name"),
                "disambiguation": release.get("disambiguation"),
            }
    return {"artist_mbid": None, "mb_name": None, "disambiguation": None}


def analyze_artist(
    name: str,
    owned_album_titles: list[str],
    releases: list[dict[str, Any]],
    owned_release_group_mbids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute completion facts for one artist from owned titles + MB releases.

    Ownership is MBID-first: an MB studio album counts as owned if its
    release-group MBID is among the owned tags, OR (fallback) its normalized
    title matches an owned album. The MBID branch is purely additive -- it can
    only turn a false "missing" into owned, never the reverse.

    Returns (summary, missing_rows). completion_pct is None when MB lists no
    studio albums for the artist (nothing to complete against).
    """
    owned_mbids = owned_release_group_mbids or set()
    studio = studio_albums(releases)
    owned_norm = {match_key(t) or normalize_title(t) for t in owned_album_titles if t}
    identity = _release_identity(releases)

    def _is_owned(key: str, rel: dict[str, Any]) -> bool:
        rg = rel.get("musicbrainz_release_group_id")
        return (bool(rg) and rg in owned_mbids) or (key in owned_norm)

    total = len(studio)
    owned_count = sum(1 for key, rel in studio.items() if _is_owned(key, rel))
    missing = [rel for key, rel in studio.items() if not _is_owned(key, rel)]
    missing.sort(key=lambda r: (int(r.get("year") or 0), normalize_title(r.get("album") or "")))

    completion = round(100.0 * owned_count / total, 1) if total else None

    summary = {
        "name": name,
        "artist_mbid": identity["artist_mbid"],
        "mb_name": identity["mb_name"],
        "disambiguation": identity["disambiguation"],
        "owned_studio": owned_count,
        "total_studio": total,
        "missing_count": len(missing),
        "completion_pct": completion,
    }
    missing_rows = [
        {
            "artist_name": name,
            "artist_mbid": identity["artist_mbid"],
            "album": rel.get("album"),
            "year": (int(rel.get("year") or 0) or None),
            "release_group_mbid": rel.get("musicbrainz_release_group_id"),
            "primary_type": rel.get("primary_type") or rel.get("release_group_type"),
            "secondary_types": rel.get("secondary_types") or [],
        }
        for rel in missing
    ]
    return summary, missing_rows


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ci_artists (
        name TEXT PRIMARY KEY,
        artist_mbid TEXT,
        mb_name TEXT,
        disambiguation TEXT,
        owned_studio INTEGER,
        total_studio INTEGER,
        missing_count INTEGER,
        completion_pct REAL,
        built_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ci_missing (
        artist_name TEXT,
        artist_mbid TEXT,
        album TEXT,
        year INTEGER,
        release_group_mbid TEXT,
        primary_type TEXT,
        secondary_types TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ci_missing_artist ON ci_missing(artist_name)")
    cur.execute("CREATE TABLE IF NOT EXISTS ci_meta (key TEXT PRIMARY KEY, value TEXT)")
    # Knowledge Layer: per-artist affinity aggregates (derived from play_history).
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ci_affinity (
        name TEXT PRIMARY KEY,
        artist_mbid TEXT,
        thumbs_up INTEGER,
        thumbs_down INTEGER,
        skips INTEGER,
        played_tracks INTEGER,
        last_played TEXT,
        recency_weight REAL,
        affinity_score REAL,
        provenance TEXT,
        built_at TEXT
    )
    """)
    # Recommendation ledger: derived SCORE rows (lifecycle lives in play_history).
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        rec_id TEXT PRIMARY KEY,
        lane TEXT,
        release_group_mbid TEXT,
        artist_mbid TEXT,
        artist_name TEXT,
        album TEXT,
        year INTEGER,
        rank INTEGER,
        score REAL,
        tier TEXT,
        components TEXT,
        provenance TEXT,
        score_reason TEXT,
        run_id TEXT,
        generated_at TEXT
    )
    """)
    conn.commit()


def load_owned_artists(index_path: Path) -> list[dict[str, Any]]:
    with open(index_path, encoding="utf-8") as handle:
        library = json.load(handle)
    return library.get("artists", []) or []


def _owned_release_group_mbids(name: str) -> set[str]:
    """Owned release-group MBIDs for an artist, read from the library DB's
    MusicBrainz file tags (read-only, defensive: empty set if the DB/column is
    absent or unpopulated, which reduces cleanly to title-only matching)."""
    import musicdb
    try:
        conn = sqlite3.connect(f"file:{Path(musicdb.DB).as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT mb_release_group_id FROM tracks "
            "WHERE artist = ? AND mb_release_group_id IS NOT NULL", (name,)
        ).fetchall()
        return {r[0] for r in rows if r[0]}
    except sqlite3.OperationalError:
        return set()  # column/table not present yet
    finally:
        conn.close()


# Columns that identify the current collection.db schema. Their absence means an
# older schema (or none) and triggers a regenerate (drop + recreate), never a
# migration (Principle 9).
_EXPECTED_ARTIST_COLS = {"artist_mbid", "mb_name", "disambiguation"}
_EXPECTED_MISSING_COLS = {"artist_mbid", "secondary_types"}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def schema_is_current(conn: sqlite3.Connection) -> bool:
    """True if the fact tables exist and carry the current identity columns."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"ci_artists", "ci_missing", "ci_meta"} <= tables:
        return False
    return (_EXPECTED_ARTIST_COLS <= _table_columns(conn, "ci_artists")
            and _EXPECTED_MISSING_COLS <= _table_columns(conn, "ci_missing"))


def _reset_fact_tables(conn: sqlite3.Connection) -> None:
    """Drop and recreate the derived fact tables (schema rev = regenerate)."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ci_artists")
    cur.execute("DROP TABLE IF EXISTS ci_missing")
    ensure_schema(conn)


def upsert_artist(
    conn: sqlite3.Connection,
    summary: dict[str, Any],
    missing_rows: list[dict[str, Any]],
    built_at: str,
) -> None:
    """Write one artist's facts atomically: replace its ci_artists row and its
    ci_missing rows in a single small transaction (delete-then-insert per artist),
    committed by the caller. A crash between artists loses only the in-flight one.
    """
    name = summary["name"]
    cur = conn.cursor()
    cur.execute("DELETE FROM ci_missing WHERE artist_name = ?", (name,))
    cur.execute(
        "INSERT OR REPLACE INTO ci_artists "
        "(name, artist_mbid, mb_name, disambiguation, "
        "owned_studio, total_studio, missing_count, completion_pct, built_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, summary["artist_mbid"], summary["mb_name"], summary["disambiguation"],
         summary["owned_studio"], summary["total_studio"],
         summary["missing_count"], summary["completion_pct"], built_at),
    )
    cur.executemany(
        "INSERT INTO ci_missing "
        "(artist_name, artist_mbid, album, year, release_group_mbid, primary_type, secondary_types) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r["artist_name"], r["artist_mbid"], r["album"], r["year"],
          r["release_group_mbid"], r["primary_type"], json.dumps(r["secondary_types"]))
         for r in missing_rows],
    )


def _prune_absent_artists(conn: sqlite3.Connection, keep_names: list[str]) -> None:
    """Drop ci_artists/ci_missing rows for artists not in ``keep_names`` (used by
    a full build to match rebuild semantics; a stale-only build keeps them)."""
    if not keep_names:
        return
    placeholders = ",".join("?" for _ in keep_names)
    conn.execute(f"DELETE FROM ci_missing WHERE artist_name NOT IN ({placeholders})", keep_names)
    conn.execute(f"DELETE FROM ci_artists WHERE name NOT IN ({placeholders})", keep_names)


def _fresh_names(conn: sqlite3.Connection, refresh_days: float) -> set[str]:
    """Names whose ci_artists row is newer than the refresh window -- a stale-only
    build skips these. Missing artists are never here, so they always refresh."""
    cutoff = (datetime.now() - timedelta(days=refresh_days)).isoformat(timespec="seconds")
    return {
        name for name, built in conn.execute("SELECT name, built_at FROM ci_artists")
        if built and built >= cutoff  # same ISO format -> lexical compare is chronological
    }


def build(
    index_path: Path = INDEX,
    db_path: Path = DB,
    fetch_fn: Callable[[str], list[dict[str, Any]]] = lookup_artist_release_groups,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    stale_only: bool = False,
    refresh_days: float = REFRESH_DAYS,
    owned_mbids_fn: Callable[[str], set[str]] = _owned_release_group_mbids,
    affinity_inputs_fn: Callable[[], dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Rebuild collection.db from music_index.json + MusicBrainz. Idempotent.

    Each artist is upserted and committed individually, so an interrupted build
    leaves every already-processed artist intact and served; re-running finishes
    the rest. A schema rev (or first build) regenerates the fact tables.

    stale_only=True refreshes only artists missing from ci_artists or older than
    ``refresh_days`` and preserves the rest; the default full build refreshes
    every library artist and prunes those no longer present.
    """
    artists = load_owned_artists(index_path)
    if limit is not None:
        artists = artists[:limit]

    built_at = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        if not schema_is_current(conn):
            _reset_fact_tables(conn)  # regenerate on schema rev / first build
        conn.commit()

        skip = _fresh_names(conn, refresh_days) if stale_only else set()
        mb_found = 0
        mb_empty = 0
        refreshed = 0
        seen_names: list[str] = []
        for i, artist in enumerate(artists, 1):
            name = artist.get("name", "")
            if not name:
                continue
            seen_names.append(name)
            if name in skip:
                continue

            owned = artist.get("albums", []) or []
            releases = fetch_fn(name) or []
            if releases:
                mb_found += 1
            else:
                mb_empty += 1

            summary, missing_rows = analyze_artist(
                name, owned, releases, owned_mbids_fn(name))
            upsert_artist(conn, summary, missing_rows, built_at)
            conn.commit()  # per-artist durability
            refreshed += 1
            log(f"[{i}/{len(artists)}] {name}: "
                f"{summary['owned_studio']}/{summary['total_studio']} owned, "
                f"{summary['missing_count']} missing")

        # A full build prunes artists no longer in the library; a stale-only build
        # must keep the ones it deliberately skipped.
        if not stale_only:
            _prune_absent_artists(conn, seen_names)
        conn.commit()

        # Knowledge Layer: materialize per-artist affinity from play_history (cheap,
        # local, no network) on every build. Deferred import avoids a hard cycle.
        if affinity_inputs_fn is None:
            import musicdb
            affinity_inputs_fn = musicdb.artist_affinity_inputs
        write_affinity(conn, compute_affinity(
            affinity_inputs_fn(), datetime.now(), built_at))
        conn.commit()

        # Collection-wide meta from the DB, so counts are correct in either mode.
        agg = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(missing_count), 0) FROM ci_artists").fetchone()
        stats = {
            "built_at": built_at,
            "mode": "stale-only" if stale_only else "full",
            "artist_count": agg[0],
            "refreshed": refreshed,
            "total_missing": agg[1],
            "mb_artists_found": mb_found,
            "mb_artists_empty": mb_empty,
        }
        conn.executemany(
            "INSERT OR REPLACE INTO ci_meta (key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in stats.items()],
        )
        conn.commit()
    finally:
        conn.close()
    return stats


# --- Read API (backs the FastAPI /collection/* endpoints) -------------------
# All readers are read-only and never crawl MusicBrainz; they only query the
# materialized collection.db built by `collector build`.

NOT_BUILT = {"ok": False, "error": "collection not built"}


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open collection.db read-only; never creates the file if it is missing."""
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def is_built(db_path: Path = DB) -> bool:
    """True only if collection.db exists and holds at least one analyzed artist."""
    if not Path(db_path).exists():
        return False
    try:
        conn = _connect_ro(db_path)
    except sqlite3.OperationalError:
        return False
    try:
        row = conn.execute("SELECT COUNT(*) FROM ci_artists").fetchone()
        return bool(row and row[0] > 0)
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _artist_summary(row: tuple) -> dict[str, Any]:
    return {
        "name": row[0],
        "owned_studio": row[1],
        "total_studio": row[2],
        "missing_count": row[3],
        "completion_pct": row[4],
    }


_ARTIST_COLS = "name, owned_studio, total_studio, missing_count, completion_pct"


def _freshness(built_at: str | None, stale_after_days: float = STALE_AFTER_DAYS
               ) -> tuple[float | None, bool]:
    """Return (age_days, stale) from a build timestamp. Unknown/unparseable
    timestamps report age None and stale True (freshness cannot be confirmed)."""
    if not built_at:
        return None, True
    try:
        built = datetime.fromisoformat(built_at)
    except ValueError:
        return None, True
    age_days = round((datetime.now() - built).total_seconds() / 86400.0, 2)
    return age_days, age_days > stale_after_days


def collection_summary(db_path: Path = DB) -> dict[str, Any]:
    if not is_built(db_path):
        return dict(NOT_BUILT)
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(f"SELECT {_ARTIST_COLS} FROM ci_artists").fetchall()
        meta = dict(conn.execute("SELECT key, value FROM ci_meta").fetchall())
    finally:
        conn.close()

    scored = [r for r in rows if (r[2] or 0) > 0]
    owned = sum(r[1] or 0 for r in scored)
    total = sum(r[2] or 0 for r in scored)
    age_days, stale = _freshness(meta.get("built_at"))
    return {
        "ok": True,
        "built_at": meta.get("built_at"),
        "age_days": age_days,
        "stale": stale,
        "artists": len(rows),
        "artists_with_mb_data": len(scored),
        "owned_studio_albums": owned,
        "total_studio_albums": total,
        "missing_studio_albums": sum(r[3] or 0 for r in rows),
        "overall_completion_pct": round(100.0 * owned / total, 1) if total else None,
        "near_complete_artists": sum(1 for r in rows if (r[3] or 0) in (1, 2)),
        "complete_artists": sum(1 for r in rows if (r[2] or 0) > 0 and (r[3] or 0) == 0),
    }


def collection_completion(db_path: Path = DB, limit: int = 50) -> dict[str, Any]:
    limit = max(0, limit)
    if not is_built(db_path):
        return dict(NOT_BUILT)
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(
            f"SELECT {_ARTIST_COLS} FROM ci_artists WHERE total_studio > 0"
        ).fetchall()
    finally:
        conn.close()
    items = [_artist_summary(r) for r in rows]
    items.sort(key=lambda a: (-(a["completion_pct"] or 0), a["missing_count"], a["name"].lower()))
    return {"ok": True, "count": len(items), "artists": items[:limit] if limit else items}


def collection_gaps(db_path: Path = DB, limit: int = 20) -> dict[str, Any]:
    limit = max(0, limit)
    if not is_built(db_path):
        return dict(NOT_BUILT)
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(
            f"SELECT {_ARTIST_COLS} FROM ci_artists WHERE total_studio > 0"
        ).fetchall()
    finally:
        conn.close()
    items = [_artist_summary(r) for r in rows]
    near = [a for a in items if a["missing_count"] in (1, 2)]
    near.sort(key=lambda a: (a["missing_count"], -(a["completion_pct"] or 0), a["name"].lower()))
    biggest = sorted(items, key=lambda a: (-a["missing_count"], -(a["completion_pct"] or 0), a["name"].lower()))
    return {
        "ok": True,
        "near_complete": near[:limit] if limit else near,
        "biggest_gaps": biggest[:limit] if limit else biggest,
    }


def collection_artist(name: str, db_path: Path = DB) -> dict[str, Any]:
    if not is_built(db_path):
        return dict(NOT_BUILT)
    conn = _connect_ro(db_path)
    try:
        row = conn.execute(
            f"SELECT {_ARTIST_COLS}, artist_mbid FROM ci_artists WHERE name = ? COLLATE NOCASE",
            (name,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "artist not found", "name": name}
        missing = conn.execute(
            "SELECT album, year, release_group_mbid, primary_type FROM ci_missing "
            "WHERE artist_name = ? COLLATE NOCASE ORDER BY year, album", (row[0],)
        ).fetchall()
    finally:
        conn.close()
    result = _artist_summary(row)
    # Additive: the resolved artist MBID (name stays the public address).
    result["artist_mbid"] = row[5]
    result["missing_albums"] = [
        {"album": m[0], "year": m[1], "release_group_mbid": m[2], "primary_type": m[3]}
        for m in missing
    ]
    return {"ok": True, "artist": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collector", description="Collector Intelligence")
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build", help="Rebuild collection.db from library + MusicBrainz")
    build_cmd.add_argument("--index", type=Path, default=INDEX)
    build_cmd.add_argument("--db", type=Path, default=DB)
    build_cmd.add_argument("--limit", type=int, default=None, help="Only analyze the first N artists")
    mode = build_cmd.add_mutually_exclusive_group()
    mode.add_argument("--stale-only", action="store_true",
                      help="Refresh only new/stale artists (default cron job)")
    mode.add_argument("--full", action="store_true",
                      help="Refresh every artist and prune removed ones (default)")
    build_cmd.add_argument("--refresh-days", type=float, default=REFRESH_DAYS,
                           help=f"Staleness window for --stale-only (default {REFRESH_DAYS:g})")

    rec_cmd = sub.add_parser("recommend", help="Generate recommendations into the ledger")
    rec_cmd.add_argument("--db", type=Path, default=DB)
    rec_cmd.add_argument("--prompt", default="essential albums for my taste")
    rec_cmd.add_argument("--count", type=int, default=50)
    rec_cmd.add_argument("--lanes", default="completion,taste_fit",
                         help="Comma-separated lanes (default completion,taste_fit)")

    args = parser.parse_args(argv)
    if args.command == "build":
        mode_label = "stale-only" if args.stale_only else "full"
        print(f"Building collection.db ({mode_label}) from {args.index} ...")
        try:
            with build_lock():
                stats = build(index_path=args.index, db_path=args.db, limit=args.limit,
                              stale_only=args.stale_only, refresh_days=args.refresh_days)
        except BuildLockError as exc:
            print(f"ERROR: {exc}")
            return 1
        print()
        print("Finished.")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"  db: {args.db}")
    elif args.command == "recommend":
        lanes = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
        taste_fit_fn = _default_taste_fit_fn(args.db) if "taste_fit" in lanes else None
        print(f"Generating recommendations (lanes: {', '.join(lanes)}) ...")
        try:
            with build_lock():
                result = generate(db_path=args.db, lanes=lanes, prompt=args.prompt,
                                  count=args.count, taste_fit_fn=taste_fit_fn)
        except BuildLockError as exc:
            print(f"ERROR: {exc}")
            return 1
        print()
        print("Finished.")
        print(f"  run_id: {result['run_id']}")
        for lane, c in result["counters"].items():
            print(f"  {lane}: {c}")
        print(f"  db: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
