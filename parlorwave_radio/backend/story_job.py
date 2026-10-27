"""Preparing music stories AHEAD of air, in the background, within a fixed
budget -- and the switch that decides whether the story path is live.

Opportunistic exactly like the lyric-semantics pass (song_context), and
stricter, because it borrows the station's ONE writer model:

  * BOUNDED. A pass prepares part of one artist: at most MAX_CHUNKS_PER_PASS
    biography windows, MAX_CALLS_PER_PASS model calls (story proposals and
    the judge together) and MAX_PASS_S of wall time, each call capped at
    CALL_TIMEOUT_S of wall time and CALL_NUM_PREDICT tokens. Passes are
    single-flight and at least PREP_INTERVAL_S apart. A long biography
    resumes where the last pass stopped (dj_memory.prep_log.next_chunk).
  * PREEMPTIBLE. Every live writer call raises
    writer_gate.live() at the transport, never waiting on anything here.
    Preparation starts a call only when writer_gate.quiet(), streams it
    (providers.ollama_generate_background), and drops the connection as soon
    as writer_gate.interrupted() -- any live call since it began. A dropped,
    timed-out or failed call stops the pass, nothing it streamed is used, the
    window is done again later, and an unjudged candidate is never stored; a
    window that fails MAX_WINDOW_FAILURES times in a row is skipped.
  * YIELDING. Before EVERY model call it asks station_busy(): a spoken
    render in flight or due within PRESSURE_HORIZON_S on ANY queue (DJ
    breaks, News intros, announcement carriers and liners are all
    dj_breaks rows), a News or announcements pass running, or the
    lyric-semantics worker holding the writer. Busy means stop now; an
    interrupted window is done again later and an unjudged candidate is
    never stored.
  * THE PRODUCTION WRITER, UNCHANGED. The configured DJ writer model and
    endpoint (llm_adapter), with the Smart DJ planner's own context window
    (smart_dj.PLANNER_NUM_CTX) so a preparation call can never make the
    runtime reload the model with a different context. No writer configured
    means no preparation: the deterministic fallback stays exactly as it is.

Music Assistant is read through story_prep's allowlisted, refresh-free
reader (allow_update_metadata=False); an artist is prepared again only when
their biography text changes.

Nothing here runs unless the story path is ACTIVE (settings.dj_stories_active:
the user preference AND the rollout gate).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

PREP_INTERVAL_S = 600.0        # at most one pass every 10 minutes
PRESSURE_HORIZON_S = 120.0     # no model call while a render is due within this
REFRESH_DAYS = 30              # re-check a prepared artist's biography
MAX_CHUNKS_PER_PASS = 2        # biography windows per pass
MAX_CALLS_PER_PASS = 10        # proposal + judge calls per pass
MAX_PASS_S = 180.0             # wall time per pass
CALL_TIMEOUT_S = 20.0          # one model call's wall time: the fallback bound
                               # on a live wait if the drop ever fails
CALL_NUM_PREDICT = 450         # completion tokens for a proposal call
PREP_MAX_STORIES = 3           # stories asked for per window (fits the above)
JUDGE_NUM_PREDICT = 200
MAX_WINDOW_FAILURES = 3        # consecutive failed calls before a window is skipped

_worker = threading.Lock()


def memory_path() -> Path:
    import paths
    return Path(paths.BASE_DIR) / "dj_memory.db"


def open_memory(path: str | Path | None = None):
    import dj_memory
    return dj_memory.DJMemory(path or memory_path())


def active() -> bool:
    try:
        import settings
        return settings.dj_stories_active()
    except Exception:  # noqa: BLE001 - the gate fails closed
        return False


def writer_usable() -> bool:
    """Is a writer model configured and usable? Without one there is nothing
    to prepare with -- and nothing to borrow from."""
    try:
        import optional_services
        return bool(optional_services.effective_writer().get("usable"))
    except Exception:  # noqa: BLE001
        return False


def station_busy(now_dt=None, db_path=None,
                 horizon_s: float = PRESSURE_HORIZON_S) -> str | None:
    """Why preparation must not use the writer right now, or None. Unreadable
    state counts as busy: this protects live speech."""
    import writer_gate
    if not writer_gate.quiet():
        return "live_writer"           # a live writer call, or one just ended
    try:
        import dj_engine
        c = dj_engine._connect(db_path)
        try:
            marks = ",".join("?" for _ in dj_engine.NON_TERMINAL_STATES)
            queues = [r[0] for r in c.execute(
                f"SELECT DISTINCT queue_id FROM dj_breaks WHERE state IN "
                f"({marks})", dj_engine.NON_TERMINAL_STATES)]
        finally:
            c.close()
        for q in queues:
            if dj_engine.render_pressure(q, now=now_dt, horizon_s=horizon_s,
                                         db_path=db_path):
                return "render_pressure"
    except Exception:  # noqa: BLE001
        return "state_unreadable"
    # Plain import statements, never a computed module name: the public
    # add-on's backend is the STATIC import closure of its entrypoints
    # (release/export.py), so a module reached only dynamically could be
    # missing from a release.
    try:
        import news_schedule
        if news_schedule._pass_lock.locked():
            return "news_pass"
    except Exception:  # noqa: BLE001
        return "news_pass"
    try:
        import announcements
        if announcements._pass_lock.locked():
            return "announcements_pass"
    except Exception:  # noqa: BLE001
        return "announcements_pass"
    try:
        import song_context
        if song_context._worker.locked():
            return "semantics_pass"
    except Exception:  # noqa: BLE001
        pass
    return None


def _writer_call(prompt: str, schema: dict, *, temperature: float,
                 num_predict: int) -> str:
    """The configured DJ writer model and endpoint (llm_adapter), one
    structured generate, with the planner's own context window."""
    import providers
    import smart_dj
    import writer_gate
    from inference import llm_adapter
    adapter = llm_adapter.default_adapter()
    role = llm_adapter.ROLE_DJ_WRITER
    model = adapter.model_override() or adapter.model_for(role)
    url, key = adapter.endpoint()
    since = writer_gate.epoch()
    if writer_gate.interrupted(since) or not writer_gate.quiet():
        raise providers.GenerationCancelled("live_writer")
    body = providers.ollama_generate_background(
        prompt, model, max_s=CALL_TIMEOUT_S, format=schema,
        url=url or None, api_key=key or None,
        options={"temperature": temperature, "num_predict": num_predict,
                 "num_ctx": smart_dj.PLANNER_NUM_CTX},
        cancel=lambda: writer_gate.interrupted(since))
    if body.get("done_reason") == "length":
        raise providers.GenerationCancelled("length")   # a cut-off reply
    return str(body.get("response") or "")


def _default_llm(prompt: str, schema: dict) -> str:
    return _writer_call(prompt, schema, temperature=0.7,
                        num_predict=CALL_NUM_PREDICT)


def _default_judge(telling: str, source: str) -> bool:
    import story_prep
    raw = _writer_call(story_prep.judge_prompt(telling, source),
                       story_prep.JUDGE_SCHEMA, temperature=0.0,
                       num_predict=JUDGE_NUM_PREDICT)
    return story_prep.judge_keeps(raw)


def pack_facts(artist: str) -> dict[str, Any]:
    """The knowledge pack's SOLID facts for one act (formed year, studio-album
    years), used only to veto a contradicting story. Read-only; empty when the
    pack is absent."""
    out: dict[str, Any] = {"artist": artist}
    try:
        import music_knowledge as mk
        conn = mk._connection()
        if conn is None:
            return out
        row = conn.execute("SELECT artist_mbid FROM artists WHERE LOWER(name)"
                           " = ?", (artist.lower(),)).fetchone()
        if not row:
            return out
        mbid = row[0]
        f = conn.execute("SELECT object_value FROM facts WHERE subject_id = ? "
                         "AND predicate = 'group_formed_year'", (mbid,)
                         ).fetchone()
        if f and str(f[0])[:4].isdigit():
            out["formed"] = int(str(f[0])[:4])
        albums: dict[str, int] = {}
        for r in conn.execute("SELECT title, first_release_date, primary_type,"
                              " secondary_types_json FROM release_groups "
                              "WHERE artist_mbid = ?", (mbid,)):
            if mk.release_kind(r[2], r[3]) == mk.KIND_STUDIO and r[1] and \
                    str(r[1])[:4].isdigit():
                y = int(str(r[1])[:4])
                albums[r[0]] = min(y, albums.get(r[0], 9999))
        out["studio_albums"] = albums
    except Exception:  # noqa: BLE001 - the pack only ever vetoes
        logger.debug("story prep: pack facts unavailable", exc_info=True)
    return out


def _next_artist(items: list[dict[str, Any]], memory, now: float
                 ) -> dict[str, Any] | None:
    """A partially prepared artist first (resume it), then never-prepared
    artists (in name order), then the one whose preparation is oldest, once
    it is due a refresh."""
    import dj_memory
    fresh = oldest = None
    for item in sorted(items, key=lambda i: str(i.get("name") or "").lower()):
        st = memory.prep_state(item.get("name"))
        if st is None:
            fresh = fresh or item
            continue
        if st["outcome"] not in dj_memory.COMPLETE_OUTCOMES:
            return item                       # partial: finish it first
        at = float(st["prepared_at"] or 0)
        if now - at >= REFRESH_DAYS * 86400 and (oldest is None
                                                 or at < oldest[0]):
            oldest = (at, item)
    return fresh or (oldest[1] if oldest else None)


class _Budget:
    """MAX_CALLS_PER_PASS / MAX_PASS_S, plus the station's own priority."""

    def __init__(self, busy_fn: Callable[[], Any], now_fn=time.monotonic):
        self.busy_fn, self.now_fn = busy_fn, now_fn
        self.t0, self.calls, self.why = now_fn(), 0, None

    def stop(self) -> bool:
        if self.why:
            return True
        import writer_gate
        if not writer_gate.quiet():
            self.why = "live_writer"
        elif self.calls >= MAX_CALLS_PER_PASS:
            self.why = "call_budget"
        elif self.now_fn() - self.t0 >= MAX_PASS_S:
            self.why = "time_budget"
        else:
            busy = self.busy_fn()
            if busy:
                self.why = str(busy) if isinstance(busy, str) else "busy"
        return bool(self.why)

    def counted(self, fn):
        def call(*a, **k):
            self.calls += 1
            return fn(*a, **k)
        return call


def prepare_one(*, api_fn: Callable[[str, dict[str, Any]], Any],
                llm_fn: Callable[[str, dict], Any] | None = None,
                judge_fn: Callable[[str, str], bool] | None = None,
                memory=None, should_stop: Callable[[], Any] | None = None,
                now: float | None = None) -> dict[str, Any]:
    """One bounded pass on ONE library artist. Returns what happened."""
    import story_prep
    now = float(now if now is not None else time.time())
    memory = memory or open_memory()
    items = story_prep.library_artists(api_fn)
    item = _next_artist(items, memory, now)
    if item is None:
        return {"outcome": "nothing_to_prepare", "artists": len(items)}
    material = story_prep.artist_material(api_fn, item)
    artist = material["artist"]
    bio = material.get("bio") or ""
    bio_hash = hashlib.sha256(bio.encode("utf-8")).hexdigest()[:16]
    if not bio:
        memory.note_prepared(artist, "", 0, 0, "no_biography", now)
        return {"outcome": "no_biography", "artist": artist}
    if memory.prepared_hash(artist) == bio_hash:
        memory.note_prepared(artist, bio_hash, 0, 0, "unchanged", now)
        return {"outcome": "unchanged", "artist": artist}
    prior = memory.prep_state(artist) or {}
    resume = (prior.get("outcome") == "partial"
              and prior.get("bio_hash") == bio_hash)
    start = int(prior.get("next_chunk") or 0) if resume else 0
    failures = int(prior.get("failures") or 0) if resume else 0
    if failures >= MAX_WINDOW_FAILURES:
        logger.warning("story prep: %s window %d failed %d times; skipped",
                       artist, start, failures)
        start, failures = start + 1, 0
    budget = _Budget(should_stop or (lambda: None))
    progress: dict[str, Any] = {}
    errors: list[str] = []

    def on_error(exc: BaseException) -> str:
        import providers
        cancelled = isinstance(exc, providers.GenerationCancelled) and \
            str(exc) in ("cancelled", "live_writer")
        errors.append("live_writer" if cancelled else type(exc).__name__)
        if not cancelled:
            logger.warning("story prep: %s call failed (%s); the window is "
                           "done again later", artist, exc)
        return "stop"
    got = story_prep.prepare_artist(
        material, budget.counted(llm_fn or _default_llm),
        pack=pack_facts(artist),
        judge_fn=budget.counted(judge_fn or _default_judge),
        should_stop=budget.stop, max_chunks=MAX_CHUNKS_PER_PASS,
        start_chunk=start, progress=progress, on_error=on_error,
        max_stories=PREP_MAX_STORIES)
    kept = 0
    for st in got:
        if st.ok and memory.add_story(st.record(), now=now):
            kept += 1
    done = bool(progress.get("done"))
    # a live interruption is not the window's fault; a timeout or a writer
    # failure is, and enough of them in a row skip it
    failed = bool(errors) and errors[-1] != "live_writer"
    if errors and budget.why is None:
        budget.why = errors[-1]
    memory.note_prepared(artist, bio_hash, len(got), kept,
                         "prepared" if done else "partial", now,
                         next_chunk=0 if done else progress.get("next_chunk", 0),
                         failures=(failures + 1 if failed and
                                   progress.get("next_chunk") == start else 0))
    logger.info("story prep: %s -- windows %d-%d of %d, %d model calls, "
                "%d proposed, %d kept%s", artist, start,
                progress.get("next_chunk", start), progress.get("chunks", 0),
                budget.calls, len(got), kept,
                "" if done else f" (resumes later: {budget.why or 'budget'})")
    return {"outcome": "prepared" if done else "partial", "artist": artist,
            "proposed": len(got), "kept": kept, "calls": budget.calls,
            "next_chunk": progress.get("next_chunk"), "stopped": budget.why}


def maybe_prepare(queue_id: str | None, now_dt=None,
                  deps: dict[str, Any] | None = None, db_path=None) -> str:
    """The background pass (dj_engine calls it at music boundaries). Never
    raises; returns a short outcome for tests."""
    deps = deps or {}
    if not deps.get("stories_active_fn", active)():
        return "inactive"
    if not deps.get("writer_usable_fn", writer_usable)():
        return "no_writer"
    if not _worker.acquire(blocking=False):
        return "busy"
    memory = None
    try:
        memory = deps.get("memory") or open_memory()
        now = time.time()
        if now - memory.last_prep_at() < PREP_INTERVAL_S:
            return "too_soon"

        def busy():
            if deps.get("pressure_fn") is not None:
                return ("render_pressure"
                        if deps["pressure_fn"](PRESSURE_HORIZON_S) else None)
            return station_busy(now_dt, db_path)
        why = busy()
        if why:
            return why if deps.get("pressure_fn") is None else "render_pressure"
        api_fn = deps.get("ma_api_fn")
        if api_fn is None:
            import ma_queue
            api_fn = ma_queue._api
        res = prepare_one(api_fn=api_fn, llm_fn=deps.get("story_llm_fn"),
                          judge_fn=deps.get("story_judge_fn"), memory=memory,
                          should_stop=busy)
        return res["outcome"]
    except Exception:  # noqa: BLE001 - preparation never costs anything else
        logger.warning("story prep pass failed (nothing affected)",
                       exc_info=True)
        return "error"
    finally:
        if memory is not None and memory is not deps.get("memory"):
            memory.close()
        _worker.release()
