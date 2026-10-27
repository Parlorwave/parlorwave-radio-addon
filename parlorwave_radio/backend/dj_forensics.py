"""Append-only forensic capture of DJ planner decisions.

WHY THIS EXISTS, given that `dj_breaks` already stores everything about a
break that AIRED. Two things a later review needs are not recoverable from it:

  * A decision to SAY NOTHING leaves no break. It is recorded as
    `planner_declined` on the still-`planned` row, and that note is
    REPLACED on each new decline, keeping a `count` and only the most
    recent reason -- a row at `count: 7` has lost six
    reasons. When the intent is finally revoked the row turns
    terminal and is reaped after RETAIN_TERMINAL_H (6 h), which is
    shorter than a night.
  * The per-decision `dj_planner ...` log line does carry every reason,
    but the add-on log is a ~100-line ring (about an hour) and does not
    survive a restart.

So the negative space -- why the station stayed quiet -- evaporates
overnight while the positive space persists. This file is that negative
space, plus enough of the positive space that one stream answers the
whole question without a join.

IT IS DIAGNOSTIC ONLY. Nothing reads it back. It never influences
scheduling, speech, novelty, the queue, the lifecycle or playback, and
every entry point is fenced so that a full disk, a permission error or a
serialization fault costs a log line and nothing else. Radio first.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import paths
import service_clock

logger = logging.getLogger("musicbrain")

SCHEMA_VERSION = "2.0"

#: One line per decision, in the state directory the add-on already
#: persists (`/addon_configs/...` on a local add-on), so it survives a
#: rebuild exactly as sessions.db does.
FILENAME = "dj_forensics.jsonl"

#: The rejected draft, clipped. A break is at most ~126 words, so this
#: never truncates a real one; it is a bound on the file, not an edit.
DRAFT_MAX = 600

#: Rotate at 16 MiB and keep two generations. A decision record is ~900
#: bytes and a busy night produces well under a hundred, so this is
#: roughly four orders of magnitude more than a review of a few nights
#: needs -- the rotation exists so the file cannot grow without bound over months,
#: not because a night could reach it.
MAX_BYTES = int(os.environ.get("DJ_FORENSICS_MAX_BYTES", 16 * 1024 * 1024))
KEEP = 2

#: Off by setting DJ_FORENSICS=0. Capture is on by default: a review whose
#: telemetry had to be switched on beforehand is a review that waits for
#: the problem to happen twice.
ENABLED = os.environ.get("DJ_FORENSICS", "1").strip().lower() not in (
    "0", "false", "off", "no")


def path() -> Path:
    return paths.BASE_DIR / FILENAME


def _rotate(p: Path) -> None:
    if p.stat().st_size < MAX_BYTES:
        return
    for i in range(KEEP - 1, 0, -1):
        older = p.with_suffix(f".jsonl.{i}")
        if older.exists():
            older.replace(p.with_suffix(f".jsonl.{i + 1}"))
    p.replace(p.with_suffix(".jsonl.1"))


def append(record: dict[str, Any]) -> None:
    """One JSON object, one line, O_APPEND. Never raises."""
    if not ENABLED:
        return
    try:
        offset = service_clock.offset()
        if offset:
            # a simulated (lab) timeline: say so on every record, so the
            # real instant is always recoverable as `at` minus the offset
            record = {**record, "clock_offset_s": offset.total_seconds()}
        p = path()
        if p.exists():
            _rotate(p)
        line = json.dumps(record, ensure_ascii=False, default=str)
        # O_APPEND: concurrent renderers interleave whole lines rather
        # than overwriting each other, and no lock is taken that could
        # ever stall a break.
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - diagnostics never cost a break
        logger.debug("forensic append failed", exc_info=True)


def _build_obligations(owed: list[str] | None = None,
                       satisfied: list[str] | None = None,
                       unmet: list[str] | None = None,
                       context: dict[str, Any] | None = None,
                       spoken: str | None = None,
                       suppression_reason: str | None = None) -> dict[str, Any]:
    if suppression_reason:
        return {
            "owed": [],
            "satisfied": [],
            "unmet": [],
            "clock_due": False,
            "weather_due": False,
            "calendar_due": False,
            "suppression_reason": suppression_reason,
        }
    ctx = context or {}
    max_words = ctx.get("max_words")
    if owed is None:
        owed = list(ctx.get("obligations_due") or [])
        if not owed:
            try:
                import required_hosting
                owed = list(required_hosting.obligations(ctx, max_words) or [])
            except Exception:
                owed = []
    if unmet is None and spoken is not None:
        try:
            import required_hosting
            unmet = list(required_hosting.unmet(spoken, ctx, max_words) or [])
        except Exception:
            unmet = []
    if satisfied is None:
        unmet_set = set(unmet or [])
        satisfied = [o for o in (owed or []) if o not in unmet_set]

    clock_due = "station_time" in (owed or []) or "time" in (owed or [])
    weather_due = "weather" in (owed or [])
    calendar_due = "calendar" in (owed or [])

    return {
        "owed": list(owed or []),
        "satisfied": list(satisfied or []),
        "unmet": list(unmet or []),
        "clock_due": clock_due,
        "weather_due": weather_due,
        "calendar_due": calendar_due,
        "suppression_reason": suppression_reason,
    }


def _build_weather(context: dict[str, Any] | None,
                   spoken: str | None = None,
                   is_suppressed: bool = False) -> dict[str, Any] | None:
    if is_suppressed:
        return {
            "status": "suppressed",
            "reason": "suppressed",
            "weather_due": False,
            "weather_aired": False,
            "temperature": None,
            "condition": None,
            "observation_age_minutes": None,
            "high_state": None,
            "overnight_low_state": None,
        }
    ctx = context or {}
    if not (ctx.get("weather_checked") or ctx.get("weather_status")
            or "weather_evaluation" in ctx or ctx.get("current_weather")
            or ctx.get("weather_context")):
        return None

    we = ctx.get("weather_evaluation") or {}
    status = ctx.get("weather_status") or we.get("status")
    reason = ctx.get("weather_reason") or we.get("reason")
    weather_due = ctx.get("weather_due")
    if weather_due is None:
        weather_due = we.get("weather_due")
    if weather_due is None:
        owed = ctx.get("obligations_due") or []
        weather_due = "weather" in owed or bool(ctx.get("weather_context"))

    weather_aired = False
    if spoken:
        try:
            import dj_context
            weather_aired = "weather" in dj_context.spoken_service_concepts(spoken, ctx)
        except Exception:
            weather_aired = False

    if weather_aired:
        status = "aired"
    elif not status:
        if weather_due:
            status = "due_offered"
        elif ctx.get("current_weather") or ctx.get("weather_context"):
            status = "available"
        else:
            status = "checked_empty"

    return {
        "status": status,
        "reason": reason,
        "weather_due": bool(weather_due),
        "weather_aired": weather_aired,
        "temperature": we.get("temperature") or ctx.get("weather_temperature"),
        "condition": we.get("condition") or ctx.get("weather_condition"),
        "observation_age_minutes": we.get("age_minutes") or ctx.get("weather_age_minutes"),
        "high_state": ctx.get("high_state"),
        "overnight_low_state": ctx.get("overnight_low_state"),
    }


def _build_calendar(context: dict[str, Any] | None,
                    spoken: str | None = None,
                    decision_action: str | None = None,
                    is_suppressed: bool = False) -> dict[str, Any] | None:
    if is_suppressed:
        return {
            "status": "suppressed",
            "reason": "suppressed",
            "calendar_due": False,
            "calendar_selected": False,
            "calendar_aired": False,
            "event_id": None,
            "timing_band": None,
            "minutes_until": None,
        }
    ctx = context or {}
    if not (ctx.get("calendar_checked") or ctx.get("calendar_status")
            or "calendar_evaluation" in ctx or ctx.get("calendar_event")):
        return None

    ce = ctx.get("calendar_evaluation") or {}
    cal_ev = ctx.get("calendar_event") or {}

    status = ctx.get("calendar_status") or ce.get("status")
    reason = ctx.get("calendar_reason") or ce.get("reason")
    calendar_due = bool(cal_ev) or (ctx.get("calendar_due") if "calendar_due" in ctx else ce.get("calendar_due", False))

    calendar_selected = False
    if decision_action == "calendar":
        calendar_selected = True

    calendar_aired = False
    if spoken and cal_ev:
        try:
            import smart_dj
            calendar_aired = bool(smart_dj.calendar_spoken(spoken, {"calendar": cal_ev}))
        except Exception:
            if cal_ev.get("title") and cal_ev["title"].lower() in spoken.lower():
                calendar_aired = True

    if calendar_aired:
        status = "aired"
    elif calendar_selected:
        status = "selected"
    elif not status:
        if calendar_due:
            status = "due_offered"
        else:
            status = "checked_empty"

    event_id = cal_ev.get("id") if isinstance(cal_ev, dict) else ce.get("event_id")
    timing_band = cal_ev.get("timing_band") if isinstance(cal_ev, dict) else ce.get("timing_band")
    minutes_until = cal_ev.get("minutes_until") if isinstance(cal_ev, dict) else ce.get("minutes_until")

    return {
        "status": status,
        "reason": reason,
        "calendar_due": bool(calendar_due),
        "calendar_selected": bool(calendar_selected),
        "calendar_aired": bool(calendar_aired),
        "event_id": event_id,
        "timing_band": timing_band,
        "minutes_until": minutes_until,
    }


def _build_planner(decision_obj: dict[str, Any],
                   context: dict[str, Any] | None = None) -> dict[str, Any]:
    d = decision_obj or {}
    ctx = context or {}
    action = d.get("action")
    attempted = d.get("attempted")
    hosting_type = attempted if action == "say_nothing" and attempted else (action or "unknown")
    decline_reason = d.get("reason") if (action == "say_nothing" or d.get("reason")) else None
    reason_codes = [decline_reason] if decline_reason else (["accepted"] if action and action != "say_nothing" else [])

    policy_gates = {
        "music_anchor_required": bool(ctx.get("pre_name") or ctx.get("post_name")),
        "knowledge_available": bool(ctx.get("knowledge_facts")),
        "weather_available": bool(ctx.get("current_weather") or ctx.get("weather_context")),
        "calendar_available": bool(ctx.get("calendar_event")),
    }

    return {
        "action": action,
        "attempted": attempted,
        "hosting_type": hosting_type,
        "decline_reason": decline_reason,
        "reason_codes": reason_codes,
        "policy_gates": policy_gates,
    }


def _build_shadow_deterministic(shadow_copy: str | None = None,
                                shadow_template_id: str | None = None,
                                shadow_facts_used: list[str] | None = None,
                                floor_aired: bool = False) -> dict[str, Any] | None:
    if not floor_aired and shadow_copy is None and shadow_template_id is None:
        return None
    return {
        "floor_aired": bool(floor_aired),
        "shadow_deterministic_copy": None if floor_aired else shadow_copy,
        "template_id": shadow_template_id,
        "facts_used": list(shadow_facts_used or []),
    }


def _build_writer_evidence(writer: dict[str, Any] | None,
                           context: dict[str, Any] | None = None,
                           spoken: str | None = None,
                           smart_draft: str | None = None,
                           smart_accepted: str | None = None) -> dict[str, Any]:
    w = writer or {}
    ctx = context or {}
    mode = w.get("mode")
    provider = w.get("provider")
    model = w.get("model")
    fallback_reason = w.get("fallback_reason") or ctx.get("planner_declined_reason")

    smart = ctx.get("smart_dj") or {}
    raw_draft = smart_draft or smart.get("draft")
    accepted_copy = smart_accepted or smart.get("text")
    guard_result = None
    if mode == "smart_planner" or (accepted_copy and accepted_copy == spoken):
        guard_result = "accepted"
    elif fallback_reason:
        guard_result = "fallback"

    return {
        "mode": mode,
        "provider": provider,
        "model": model,
        "raw_smart_draft": raw_draft,
        "accepted_smart_copy": accepted_copy,
        "guard_result": guard_result,
        "fallback_reason": fallback_reason,
        "final_aired_copy": spoken,
    }


def decision(token: str, queue_id: str, boundary: str,
             decision_obj: dict[str, Any], context: dict[str, Any] | None,
             now: datetime) -> None:
    """The planner's answer at one boundary -- INCLUDING silence.

    Written the moment the decision is validated, before any render, so a
    break that is never produced still leaves a record of why."""
    try:
        ctx = context or {}
        d = decision_obj or {}
        offered = []
        for f in (ctx.get("knowledge_facts") or []):
            row = {"id": f.get("id"), "fact_id": f.get("fact_id"),
                   "predicate": f.get("predicate"),
                   "subject_type": f.get("subject_type"),
                   # WHY this fact was allowed into the packet at all
                   # (fact_relevance). A review that can see the reason can
                   # tell a fact that earned its place from one that
                   # merely existed.
                   "relation": f.get("relation")}
            if f.get("kind") == "bridge":
                # "Why did the DJ think these two things belonged
                # together?" -- the relationship type, its values and the
                # pack fact_ids that are its evidence (fact_editorial).
                row["bridge_type"] = f.get("bridge_type")
                row["evidence"] = list(f.get("evidence") or [])
                row["values"] = dict(f.get("values") or {})
            offered.append(row)
        kdiag = ctx.get("knowledge_diag") or {}
        suppressed_facts = list(kdiag.get("suppressed") or [])
        editorial = kdiag.get("editorial") or {}
        cal = ctx.get("calendar_event") or {}
        rec = {
            "at": now.isoformat(timespec="seconds"),
            "schema_version": SCHEMA_VERSION,
            "event": "planner_decision",
            "break_id": token,
            "queue_id": queue_id,
            "boundary": boundary,
            "action": d.get("action"),
            "reason": d.get("reason"),
            "attempted": d.get("attempted"),
            "guard_category": d.get("guard_category"),
            # warn-only observations on an ACCEPTED break
            # (writer_guard.warnings): counted here, never a reason the
            # break did not air
            "warnings": list(d.get("warnings") or []),
            "markup": d.get("markup"),
            "claim": d.get("claim"),
            "topic_key": d.get("topic_key"),
            "fact_ids_cited": list(d.get("fact_ids") or []),
            # supplied facts the copy realized without citing; the
            # validator inferred the citation (smart_dj.infer_citations)
            "fact_ids_inferred": list(d.get("inferred_fact_ids") or []),
            "knowledge_offered": offered,
            # TRUE MATERIAL THE STATION HELD BACK. The negative space of
            # the relevance layer: without it a review can see what aired
            # and never what was deliberately not offered.
            "knowledge_suppressed": suppressed_facts,
            # THE EDITORIAL LEDGER: raw candidate counts, every derived
            # relationship (offered or withheld, with the reason), and the
            # speakability classes that held facts back. Bounded upstream.
            "knowledge_editorial": editorial if editorial else None,
            # validator-only grounding rows (never shown to the writer)
            "knowledge_reference": [{"id": f.get("id"),
                                     "predicate": f.get("predicate")}
                                    for f in (ctx.get("knowledge_reference")
                                              or [])],
            "candidates": list(d.get("candidates") or []),
            "packet_bytes": d.get("packet_bytes"),
            "chattiness": ctx.get("chattiness"),
            "persona": ctx.get("persona_profile"),
            "max_words": ctx.get("max_words"),
            "ending": ctx.get("pre_name"),
            "next": ctx.get("post_name"),
            "context_offered": [k for k in
                                ("current_weather", "weather_context",
                                 "upcoming_event", "news_headline",
                                 "knowledge_facts") if ctx.get(k)],
            "calendar_event": cal.get("title") or None,
            "calendar_band": cal.get("timing_band") or None,
            # the words themselves, when there were any
            "text": d.get("text"),
            "word_count": len((d.get("text") or "").split()) or None,
        }
        # THE REJECTED DRAFT. Without it a long run could say
        # how many decisions died on
        # `unauthorized_transition_relationship` and not one word about
        # WHY -- a category name with no evidence under it. A rejected
        # draft never aired and never will; it is kept so the guard that
        # refused it can be judged.
        draft = d.get("draft")
        if draft:
            rec["draft"] = str(draft)[:DRAFT_MAX]
            rec["draft_word_count"] = len(str(draft).split())
        for k in ("fact_id", "predicate", "leak", "clock"):
            if d.get(k):
                rec[k] = d[k]
        # A citation the packet never supplied: WHICH ids the model made
        # up, and how many facts it had actually been offered. Bounded at
        # the validator (smart_dj._bounded_ids); ids are opaque keys.
        if "unknown_fact_ids" in d:
            rec["unknown_fact_ids"] = list(d.get("unknown_fact_ids") or [])
            rec["offered_fact_count"] = d.get("offered_fact_count")
        # A refused line still has to say WHICH facts were in play. The
        # decision's own `fact_ids` is empty on every rejection (nothing
        # aired, so nothing may be marked used), which is right for
        # novelty and useless for diagnosis -- so the real ids travel on
        # their own key.
        if d.get("cited_fact_ids"):
            rec["cited_fact_ids"] = list(d["cited_fact_ids"])
        # A supplied reading the writer filed under `fact_ids` by its field
        # name ("time", "weather"): not a citation, dropped by the validator.
        if d.get("moment_refs"):
            rec["moment_refs"] = list(d["moment_refs"])
        # WHICH WRITER. Present only when the opt-in P2 writer
        # (DJ_SMART_WRITER=p2) made this decision; a row without it is the
        # production structured planner. `generation` is the model's own
        # evidence (done_reason, token counts, timing) and `stage` the
        # spoken stage direction a P2 draft was refused for.
        if d.get("writer_style"):
            rec["writer_style"] = d["writer_style"]
            if d.get("generation"):
                rec["generation"] = dict(d["generation"])
            if d.get("stage"):
                rec["stage"] = d["stage"]
            # P2 retries: every attempt's verdict and (refused) draft,
            # and whether the one content retry ran, was skipped, and why.
            if d.get("attempts"):
                rec["attempts"] = [dict(a) for a in d["attempts"]][:2]
            if d.get("retry"):
                rec["retry"] = dict(d["retry"])
            if d.get("accepted_attempt"):
                rec["accepted_attempt"] = d["accepted_attempt"]

        # Schema 2.0 additions:
        rec["planner"] = _build_planner(d, ctx)
        rec["obligations"] = _build_obligations(context=ctx)
        wx = _build_weather(ctx)
        if wx is not None:
            rec["weather"] = wx
        cal_block = _build_calendar(ctx, decision_action=d.get("action"))
        if cal_block is not None:
            rec["calendar"] = cal_block
        rec["knowledge"] = {
            "offered_ids": [f.get("id") or f.get("fact_id") for f in (ctx.get("knowledge_facts") or []) if (f.get("id") or f.get("fact_id"))],
            "cited_ids": list(d.get("fact_ids") or []),
            "suppressed": suppressed_facts,
        }

        append(rec)
    except Exception:  # noqa: BLE001
        logger.debug("forensic decision record failed", exc_info=True)


LIFECYCLE_RENDERED = "rendered"
LIFECYCLE_COMPLETED = "completed"

#: The render-time record keeps `event: "aired"` for compatibility -- every
#: existing reader filters on it -- but it is NOT proof of air, and the
#: `lifecycle` field now says so explicitly. Its PRESENCE is also the marker
#: that separates new rows from old: a row without `lifecycle` was written
#: by an older version and carries the old ambiguous meaning. Old rows are
#: never reinterpreted.
EVENT_AIRED = "aired"
EVENT_COMPLETED = "completed"


def completed(token: str, now: datetime,
              queue_id: str | None = None) -> None:
    """The break REACHED AIR: a real music track started after it, so the
    lifecycle says it played rather than that it was written."""
    try:
        rec = {"at": now.isoformat(timespec="seconds"),
               "schema_version": SCHEMA_VERSION,
               "event": EVENT_COMPLETED,
               "lifecycle": LIFECYCLE_COMPLETED,
               "break_id": token}
        if queue_id:
            rec["queue_id"] = queue_id
        append(rec)
    except Exception:  # noqa: BLE001
        logger.debug("forensic completed record failed", exc_info=True)


def aired(token: str, spoken: str, context: dict[str, Any] | None,
          writer: dict[str, Any] | None, realized: list[str] | None,
          offered_topics: list[str] | None,
          knowledge_aired: list[str] | None, now: datetime,
          hosting: dict[str, Any] | None = None,
          callback: dict[str, Any] | None = None,
          owed: list[str] | None = None,
          knowledge_realized: list[str] | None = None,
          shadow_copy: str | None = None,
          shadow_template_id: str | None = None,
          shadow_facts_used: list[str] | None = None,
          floor_aired: bool = False,
          callback_obj: dict[str, Any] | None = None,
          smart_draft: str | None = None,
          smart_accepted: str | None = None,
          voice: dict[str, Any] | None = None) -> None:
    """The copy that was ACCEPTED FOR AIR, and what it realized.

    NOT PROOF OF AIR. This is written after the accepted copy exists --
    past the writer, the repair and the callback composition -- but before
    TTS, before the insert and before anything could revoke the break. The
    `event` name stays "aired" so existing readers keep working; the
    `lifecycle: "rendered"` field is the truthful one, and a matching
    `completed` record is what proves the listener heard it."""
    try:
        ctx = context or {}
        cal = ctx.get("calendar_event") or {}
        rec = {
            "at": now.isoformat(timespec="seconds"),
            "schema_version": SCHEMA_VERSION,
            "event": EVENT_AIRED,
            # The truthful name for this moment. A reader that sees it
            # knows the row is render-time and that `completed` is the
            # proof-of-air; a row WITHOUT it predates this field.
            "lifecycle": LIFECYCLE_RENDERED,
            "break_id": token,
            "text": spoken,
            "word_count": len((spoken or "").split()),
            "writer": writer or {},
            "context_offered": list(offered_topics or []),
            "context_topics": list(realized or []),
            # cited AND audible -- the stricter test, and the only one the
            # novelty ledger is allowed to spend from
            "knowledge_aired": list(knowledge_aired or []),
            # audible only, no citation required: what the DETERMINISTIC
            # floor put on air is visible here and nowhere else
            "knowledge_realized": list(knowledge_realized or []),
            # the obligation set for this opportunity, ALWAYS -- not only
            # when the floor had to step in
            "required_obligations": list(owed or []),
            "calendar_event": cal.get("title") or None,
            "calendar_band": cal.get("timing_band") or None,
            "chattiness": ctx.get("chattiness"),
            "persona": ctx.get("persona_profile"),
            # WHO ENDED UP HOSTING, and why the model did not.
            "writer_mode": (writer or {}).get("mode"),
        }
        if hosting:
            rec["required_hosting"] = dict(hosting)
        if callback:
            rec["callback"] = dict(callback)

        # Schema 2.0 additions:
        rec["obligations"] = _build_obligations(
            owed=owed or (hosting.get("obligations") if hosting else None),
            unmet=hosting.get("unmet") if hosting else None,
            context=ctx,
            spoken=spoken,
        )
        wx = _build_weather(ctx, spoken=spoken)
        if wx is not None:
            rec["weather"] = wx
        cal_block = _build_calendar(ctx, spoken=spoken)
        if cal_block is not None:
            rec["calendar"] = cal_block

        a_ids = list(knowledge_aired or [])
        r_ids = list(knowledge_realized or [])
        rec["knowledge"] = {
            "aired_ids": a_ids,
            "realized_ids": r_ids,
            "knowledge_aired": bool(a_ids or r_ids),
        }

        cb_data = callback or ctx.get("callback_decision")
        if cb_data:
            candidate = cb_data.get("candidate") or cb_data.get("artist")
            selected = bool(cb_data.get("selected"))
            cb_aired = False
            if selected:
                if callback_obj is not None:
                    cb_aired = True
                elif spoken and candidate and candidate.lower() in spoken.lower():
                    cb_aired = True
            rec["callback_evidence"] = {
                "candidate": candidate,
                "selected": selected,
                "reason": cb_data.get("reason"),
                "suppression_reason": cb_data.get("suppression_reason"),
                "recency": cb_data.get("recency"),
                "callback_aired": cb_aired,
            }

        rec["shadow_deterministic"] = _build_shadow_deterministic(
            shadow_copy=shadow_copy,
            shadow_template_id=shadow_template_id,
            shadow_facts_used=shadow_facts_used,
            floor_aired=floor_aired,
        )

        rec["writer_evidence"] = _build_writer_evidence(
            writer=writer,
            context=ctx,
            spoken=spoken,
            smart_draft=smart_draft,
            smart_accepted=smart_accepted,
        )

        if voice:
            # Which DJ voice this render was routed to -- the row's
            # voice_id / tts_provider, and DJ Voice Switching's
            # voice_check when the owner's selection was set aside -- so a
            # later read after the row is reaped still says who spoke.
            rec["voice"] = dict(voice)

        # WHICH WRITER'S WORDS ARE IN THIS RENDER. Only on a break the
        # opt-in P2 writer planned (absent = production planner): the
        # writer mode, and whose copy actually went to TTS -- the P2 draft,
        # or the deterministic floor that took the slot after a refusal.
        if ctx.get("writer_style"):
            rec["writer_style"] = ctx["writer_style"]
            rec["aired_writer"] = ("floor" if floor_aired
                                   else ctx["writer_style"])

        append(rec)
    except Exception:  # noqa: BLE001
        logger.debug("forensic aired record failed", exc_info=True)


EVENT_SUPPRESSED = "opportunity_suppressed"
LIFECYCLE_SUPPRESSED = "suppressed"


def suppressed(token: str, queue_id: str, reason: str, purpose: str,
               state_before: str | None, target_air_at: str | None,
               now: datetime) -> None:
    """An ORDINARY opportunity the station deliberately did not speak.

    Today the only reason is `dj_speech_disabled` (settings.dj.
    speech_enabled off). It is its own event, not a planner decision and
    not a render row, so no reader can count it as a decline, a writer or
    TTS failure, an insert failure, or something the listener heard.

    `delivered` is always False and `obligations_evaluated` always False:
    the closing does not gather context, so this record does not claim to
    know which clock / weather / calendar obligations the slot would have
    carried -- only that none of them was paid by anything the listener
    heard here. `state_before` says whether the slot closed before any
    render (`planned`, the ordinary case: no ledger was touched) or after
    one (`ready`: a render row may exist beside this record, and it was
    never heard)."""
    try:
        append({
            "at": now.isoformat(timespec="seconds"),
            "schema_version": SCHEMA_VERSION,
            "event": EVENT_SUPPRESSED,
            "lifecycle": LIFECYCLE_SUPPRESSED,
            "break_id": token,
            "queue_id": queue_id,
            "reason": reason,
            "purpose": purpose,
            "state_before": state_before,
            "target_air_at": target_air_at,
            "delivered": False,
            "obligations_evaluated": False,
            "obligations": _build_obligations(suppression_reason=reason),
            "weather": _build_weather(None, is_suppressed=True),
            "calendar": _build_calendar(None, is_suppressed=True),
            "shadow_deterministic": None,
        })
    except Exception:  # noqa: BLE001
        logger.debug("forensic suppressed record failed", exc_info=True)
