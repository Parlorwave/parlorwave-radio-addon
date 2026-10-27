"""Broadcast Diagnostics: a READ-ONLY projection of what the station
already records about each DJ opportunity -- why it aired as its writer
wrote it, as the station floor's own line, or not at all.

SOURCES. Nothing else is read; nothing is written; no new ledger.
  dj_forensics.jsonl (+ its rotated generations) -- `planner_decision`,
      `aired` (RENDER time, not proof of air), `insert`, `completed` (proof
      of air) and `opportunity_suppressed`. Read NEWEST FIRST in blocks,
      within SCAN_BUDGET_BYTES per request, never loaded whole.
  the break's dj_breaks row (dj_engine) -- state, the revoke reason
      (`error`), duration, and telemetry (air_start, coverage, purpose,
      writer_authority_guard). Only while the row lives: the reaper retires
      completed rows after 24 h and other terminal rows after 6 h.

HONESTY RULES.
  * Absent evidence is `None` ("NOT RECORDED" / "UNKNOWN" in the UI), never
    a guess; `aired_to_room` is False only on a recorded negative
    (suppression, revoked / missed / failed row, a terminal insert
    failure).
  * An offending span is reported only when the refusing check recorded
    one (`planner_decision.leak` / `.claim` / `.clock`); never inferred
    from the draft.
  * Two record shapes are read as they are. Before the record shape tag
    "9dd863e" (a fixed format identifier, written as `record_shape`)
    `aired.obligations.unmet` describes the refused DRAFT whenever a
    required-hosting note exists, so it is not used as the aired verdict
    then; with that shape it describes the aired words (`draft_unmet` beside
    it) and `completed` carries `delivered` + `coverage`.

NEVER IN THE OUTPUT: the 32-hex break token (the audio route's
credential: an opportunity is `now_playing.program_ref("dj_break", token)`)
or any part of it (no `break_ref`), audio paths, URLs, calendar
titles (masked in drafts and aired copy too, however their words are spaced,
and never inside a longer word), announcement messages (no record names them,
so every free-text field of a break that carried one is withheld, the insert
detail included). Every response is built from an allowlist and scrubbed once
more on the way out.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Iterator

import dj_engine
import dj_forensics
import now_playing

SCHEMA = "broadcast_diagnostics.v1"
#: Most forensic bytes one request reads (a busy station writes ~70 KB/h: ~5 days).
SCAN_BUDGET_BYTES = int(os.environ.get("DJ_DIAGNOSTICS_SCAN_BYTES", 8 * 1024 * 1024))
#: A break's records span minutes (a quarter hour at most); once the scan is an
#: hour older than a break's newest record, its oldest has been read.
SETTLE_S = 3600
LIMIT_MAX = 200
_BLOCK = 64 * 1024

_TOKEN = re.compile(r"^[a-f0-9]{32}$")
_REF = re.compile(r"^[a-f0-9]{16}$")
_CODE = re.compile(r"[a-z][a-z0-9_]*")
_EVENTS = {"planner_decision", "aired", "insert", "completed", "opportunity_suppressed"}
OUTCOMES = ("accepted", "fallback", "rejected", "suppressed", "delivery_failure", "program", "unknown")
KINDS = ("dj_break", "news_intro", "announcement", "all")
OBLIGATION_KEYS = ("station_time", "weather", "calendar")
#: stands in for every free-text field of a break that carried an announcement
ANNOUNCEMENT_WITHHELD = "[announcement withheld]"

#: revocations that are station policy (another program owns the slot), not a
#: delivery fault: shown as `suppressed`, with the exact reason
_POLICY_REVOKES = {"news_boundary_owned", "announcement_boundary_owned", "station_change"}
#: writer modes whose aired copy is the installation's own writer's, as
#: written: the Smart DJ planner, a free-text writing model, a prepared Music
#: Story, and the built-in writer of an installation with no writing model
#: (the public Alpha's default). Only deterministic_fallback -- a writer that
#: failed, and the floor that hosted instead -- is a fallback.
_AS_WRITTEN = {"smart_planner", "ollama_primary", "story_prepared", "deterministic_primary"}
_PLANNER_OWN = {"model_said_nothing", "no_material"}
_GENERATION = {"timeout", "provider_error", "invalid_result", "truncated_generation"}
_INSERT_FAILED = {"insert_rejected", "successor_mismatch", "head_changed", "ma_api_failure", "lifecycle_unsafe"}
#: the deterministic floor's fact names -> the obligation they pay
_FLOOR_FACTS = {"ending": "ending_track", "next": "next_track", "station_time": "station_time",
                "current_weather": "weather", "calendar_event": "calendar"}
_HEADLINES = {
    "ungrounded_claim": "A fact-shaped claim the packet did not supply",
    "unsupplied_clock": "A clock time the packet did not supply",
    "internal_metadata_leak": "Internal metadata in the copy",
    "guard_rejected": "Writer authority guard rejected the copy",
    "fact_reference_mismatch": "Cited fact ids did not match the copy",
    "fact_value_not_said": "A cited fact's value was not said",
    "fact_subject_not_said": "A cited fact's subject was not said",
    "fact_relationship_drift": "A cited relationship was restated differently",
    "required_hosting_unmet": "Owed hosting (tracks, time, weather) not met",
    "model_said_nothing": "The model chose silence",
    "no_material": "Nothing to host",
    "timeout": "The writer timed out",
    "provider_error": "The writer service failed",
    "dj_speech_disabled": "DJ speech is turned off",
    "news_boundary_owned": "News owned this boundary",
    "announcement_boundary_owned": "An announcement owned this boundary",
    "stale_context": "The queue moved on before air",
    "stale_pre_air": "The break went stale before air",
    "too_late": "Too late for its boundary",
    "head_changed": "The playing item changed before insertion",
    "insert_rejected": "Insertion refused before air",
    "cut": "Cut short on air",
}

_SCRUB = [(re.compile(r"\b[a-f0-9]{32}\b"), "[redacted]"),
          (re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I), "[redacted-url]"),
          (re.compile(r"(?<![\w.])/(?:data|config|addon_configs|share|ssl|media|mnt|tmp|root|app)/\S*"),
           "[redacted-path]")]


# --- reading -----------------------------------------------------------------------

def _stats() -> dict[str, Any]:
    return {"bytes_read": 0, "budget_bytes": SCAN_BUDGET_BYTES, "budget_exhausted": False,
            "complete": False, "sources": [], "unreadable_lines": 0, "oldest_at": None,
            "seen": set()}


def _lines_newest_first(stats: dict[str, Any]) -> Iterator[bytes]:
    """The stream's lines, newest first: the live file, then its rotations,
    read backwards in blocks until the budget is spent."""
    base = dj_forensics.path()
    for p in [base] + [base.with_suffix(f".jsonl.{i}") for i in range(1, dj_forensics.KEEP + 1)]:
        try:
            fh = open(p, "rb")
        except OSError:
            continue
        with fh:
            stats["sources"].append(p.name)
            pos, tail = fh.seek(0, os.SEEK_END), b""
            while pos > 0:
                if stats["bytes_read"] >= stats["budget_bytes"]:
                    stats["budget_exhausted"] = True
                    return
                step = min(_BLOCK, pos)
                pos -= step
                fh.seek(pos)
                lines = (fh.read(step) + tail).split(b"\n")
                stats["bytes_read"] += step
                tail = lines.pop(0)          # may be cut: completed by the next block
                yield from (ln for ln in reversed(lines) if ln.strip())
            if tail.strip():
                yield tail
    stats["complete"] = True


def _records(stats: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for line in _lines_newest_first(stats):
        try:
            r = json.loads(line)
        except ValueError:
            stats["unreadable_lines"] += 1
            continue
        if isinstance(r, dict) and r.get("event") in _EVENTS \
                and _TOKEN.match(str(r.get("break_id") or "")) and isinstance(r.get("at"), str):
            if stats["oldest_at"] is None or r["at"] < stats["oldest_at"]:
                stats["oldest_at"] = r["at"]
            stats["seen"].add(r["break_id"])
            yield r


def _seconds(a: str, b: str) -> float:
    try:
        return (datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()
    except ValueError:
        return 0.0


def _groups(stats: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    """One list of records per break (newest first), breaks in order of their
    newest record. A break is released once the scan is SETTLE_S older than
    that record, or when the scan ends; memory holds only that window."""
    open_: dict[str, list[dict[str, Any]]] = {}
    released: set[str] = set()
    for r in _records(stats):
        while open_:
            first = next(iter(open_))
            if _seconds(open_[first][0]["at"], r["at"]) <= SETTLE_S:
                break
            released.add(first)
            yield open_.pop(first)
        if r["break_id"] not in released:
            open_.setdefault(r["break_id"], []).append(r)
    yield from open_.values()


class _Rows:
    """dj_breaks rows on ONE connection per request (read-only queries)."""

    def __init__(self, db_path=None):
        self.c = dj_engine._connect(db_path)

    def get(self, token: str) -> dict[str, Any] | None:
        row = self.c.execute(f"SELECT {', '.join(dj_engine._COLS)} FROM dj_breaks WHERE token = ?",
                             (token,)).fetchone()
        if row is None:
            return None
        d = dict(zip(dj_engine._COLS, row))
        try:
            d["telemetry"] = json.loads(d.get("telemetry") or "{}") or {}
        except (TypeError, ValueError):
            d["telemetry"] = {}
        return d

    def tokens(self) -> dict[str, str]:
        """{token: created_at} of every row still held (bounded by retention)."""
        return {t: at for t, at in self.c.execute("SELECT token, created_at FROM dj_breaks")}

    def close(self) -> None:
        self.c.close()


# --- projection --------------------------------------------------------------------

def _ref(token: str) -> str:
    return now_playing.program_ref("dj_break", token) or ""


def _kind(aired: dict | None, tel: dict) -> str:
    purpose, mode = tel.get("purpose"), str((aired or {}).get("writer_mode") or "")
    if purpose == "news_intro" or mode == "template_line":
        return "news_intro"
    if purpose == "announcement" or mode.startswith("announcement"):
        return "announcement"
    return "dj_break"


def _delivery(row: dict | None, completed: dict | None, inserts: list[dict]) -> dict[str, Any]:
    """Did the room hear it? Only recorded evidence decides."""
    tel = (row or {}).get("telemetry") or {}
    state = (row or {}).get("state")
    if completed or state == "completed":
        if completed and "coverage" in completed:
            cov, src = completed["coverage"], "completed.coverage"
        elif row is not None:
            cov, src = (tel.get("coverage") or {}).get("verdict") or "whole", "dj_breaks.telemetry.coverage"
        else:
            cov, src = None, None
        return {"state": "cut" if cov == "cut" else "completed", "aired_to_room": True, "reason": None,
                "coverage": cov, "coverage_source": src,
                "source": "completed" if completed else "dj_breaks.state"}
    if state in ("revoked", "missed", "failed"):
        return {"state": state, "aired_to_room": False, "reason": row.get("error") or None,
                "coverage": None, "coverage_source": None, "source": "dj_breaks.state"}
    if state:                       # still live (a replan may yet air it)
        return {"state": "pending", "aired_to_room": None, "reason": None, "coverage": None,
                "coverage_source": None, "source": "dj_breaks.state"}
    terminal = [i for i in inserts if i.get("outcome") in _INSERT_FAILED | {"insert_success"}]
    if terminal and terminal[-1]["outcome"] != "insert_success":
        return {"state": "insert_failed", "aired_to_room": False, "reason": terminal[-1]["outcome"],
                "coverage": None, "coverage_source": None, "source": "insert.outcome"}
    return {"state": None, "aired_to_room": None, "reason": None, "coverage": None,
            "coverage_source": None, "source": None}


def _stage_for_planner(reason: str | None) -> str:
    if reason in _PLANNER_OWN:
        return "planner_decision"
    if reason in _GENERATION:
        return "writer_generation"
    return "planner_guard" if reason == "guard_rejected" else "planner_validate"


def _span(dec: dict | None) -> tuple[str | None, str | None]:
    for k in ("leak", "claim", "clock"):
        v = (dec or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"planner_decision.{k}"
    return None, None


def _why(outcome: str, ev: dict[str, Any]) -> dict[str, Any]:
    """The one recorded reason that explains the outcome, with where it is
    written. Nothing is synthesised: no record, no reason."""
    dec, aired, sup, deliv, tel = ev["decision"], ev["aired"], ev["suppressed"], ev["delivery"], ev["tel"]
    stage = code = source = guard = None
    if sup:
        stage, code, source = "pre_planner_gate", sup.get("reason"), "opportunity_suppressed.reason"
    elif outcome in ("suppressed", "delivery_failure"):
        code, source = deliv["reason"], ("dj_breaks.error" if deliv["source"] == "dj_breaks.state"
                                         else deliv["source"])
        if deliv["state"] == "cut":
            stage, code, source = "on_air_playthrough", "cut", deliv["coverage_source"]
        else:
            stage = "program_arbitration" if code in _POLICY_REVOKES else "queue_insert_gate"
    elif dec and dec.get("action") == "say_nothing":
        code, source, guard = dec.get("reason"), "planner_decision.reason", dec.get("guard_category")
        stage = _stage_for_planner(code)
    elif aired and (aired.get("writer") or {}).get("fallback_reason"):
        code, source = aired["writer"]["fallback_reason"], "aired.writer.fallback_reason"
        wag = tel.get("writer_authority_guard") or {}
        stage = {"required_hosting_unmet": "render_required_hosting",
                 "guard_rejected": "render_writer_guard"}.get(code, "render_writer")
        guard = wag.get("violation_category") if code == "guard_rejected" else None
    elif outcome == "accepted" and deliv["state"] == "completed":
        stage = "passed_all"
    out = {"rejected": outcome in ("fallback", "rejected", "suppressed", "delivery_failure"),
           "validation_stage": stage, "reason_code": code, "reason_source": source,
           "guard_category": guard, "headline": _HEADLINES.get(code or "", code)}
    return out


def _owed(ev: dict[str, Any]) -> list[str] | None:
    for r, path in ((ev["aired"], "required_obligations"), (ev["aired"], "obligations"),
                    (ev["decision"], "obligations")):
        if not r:
            continue
        v = r.get(path)
        v = v.get("owed") if isinstance(v, dict) else v
        if isinstance(v, list) and (v or path != "required_obligations"):
            return ["station_time" if k == "time" else k for k in v]
    return None


def _in_aired(ev: dict[str, Any]) -> tuple[dict[str, bool | None], str | None, bool]:
    """{obligation: was it in the AIRED words}, where that is written, and
    whether the record is the newer shape (`draft_unmet` present)."""
    aired = ev["aired"]
    if not aired:
        return {}, None, False
    obl = aired.get("obligations") or {}
    hosting = aired.get("required_hosting") or {}
    owed = _owed(ev) or []
    new_shape = "draft_unmet" in obl
    if new_shape or hosting.get("unmet") is None:
        unmet = set(obl.get("unmet") or [])
        return {k: k not in unmet for k in owed}, "aired.obligations.unmet", new_shape
    # older shape with a hosting note: obligations.unmet is the DRAFT's
    if aired.get("writer_mode") == "deterministic_fallback":
        used = {_FLOOR_FACTS.get(f) for f in (aired.get("writer") or {}).get("required_facts_used") or []}
        return {k: (True if k in used else None) for k in owed}, "aired.writer.required_facts_used", False
    comp = hosting.get("completion")
    if isinstance(comp, dict) and comp.get("completed"):
        draft_unmet, added = set(hosting.get("unmet") or []), set(comp.get("missing") or [])
        return ({k: (True if (k not in draft_unmet or k in added) else None) for k in owed},
                "aired.required_hosting.unmet + completion", False)
    return {k: None for k in owed}, None, False


def _obligations(ev: dict[str, Any], deliv: dict[str, Any]) -> dict[str, Any]:
    aired, comp = ev["aired"], ev["completed"]
    owed = _owed(ev)
    said, said_src, new_shape = _in_aired(ev)
    srcs = {k: said_src for k in said}
    for key, block, flag in (("weather", "weather", "weather_aired"),
                             ("calendar", "calendar", "calendar_aired")):
        b = (aired or {}).get(block)
        if isinstance(b, dict) and flag in b and owed and key in owed:
            said[key] = bool(b[flag])           # judged on the spoken words
            srcs[key] = f"aired.{block}.{flag}"
    hosting = (aired or {}).get("required_hosting") or {}
    draft_unmet = ((aired or {}).get("obligations") or {}).get("draft_unmet")
    if draft_unmet is None and hosting.get("unmet") is not None:
        draft_unmet = hosting.get("unmet")
    delivered_list = comp.get("delivered") if comp and "delivered" in comp else None
    items = []
    for key in OBLIGATION_KEYS:
        due = None if owed is None else key in owed
        in_aired = said.get(key) if due else None
        dsrc = (f"delivery {deliv['state']} ({deliv['source']})" if deliv["state"]
                else "no delivery record in the history read")
        if due is False:
            delivered, dsrc = "not_due", None
        elif due is None:
            delivered, dsrc = None, None
        elif deliv["aired_to_room"] is False:
            delivered = "not_delivered"
        elif deliv["state"] == "cut":
            delivered = "cut"
        elif deliv["state"] != "completed":
            delivered = None
        elif delivered_list is not None and key in ("station_time", "weather"):
            delivered, dsrc = ("confirmed" if key in delivered_list else "not_delivered"), "completed.delivered"
        elif in_aired is False:
            delivered = "not_delivered"
        elif in_aired and deliv["coverage"] == "whole":
            delivered, dsrc = "confirmed", f"completed; coverage from {deliv['coverage_source']}"
        elif in_aired:
            delivered, dsrc = "aired_coverage_not_recorded", "completed; coverage not recorded (row retired)"
        else:
            delivered = None
        note = "; ".join(x for x in (f"in_aired: {srcs[key]}" if due and srcs.get(key) else None,
                                     f"delivered: {dsrc}" if dsrc else None) if x) or None
        items.append({"key": key, "due": due,
                      "in_draft": (None if not due or draft_unmet is None else key not in draft_unmet),
                      "in_aired": in_aired, "delivered": delivered, "note": note})
    return {"owed": owed, "satisfied_in_aired": [k for k, v in said.items() if v] if said_src else None,
            "satisfied_source": said_src, "unmet_in_draft": draft_unmet,
            "confirmed_delivered": [i["key"] for i in items if i["delivered"] == "confirmed"]
            if owed is not None else None,
            "record_shape": "9dd863e" if (new_shape or delivered_list is not None) else
                            ("pre_9dd863e" if aired or comp else None),
            "items": items}


def _evidence(recs: list[dict[str, Any]], row: dict | None) -> dict[str, Any]:
    newest_first = list(recs)

    def last(event):
        return next((r for r in newest_first if r["event"] == event), None)
    decisions = [r for r in reversed(newest_first) if r["event"] == "planner_decision"]
    aired = last("aired")
    if aired:   # the decision that led to this render, not a later replan
        decisions = [d for d in decisions if d["at"] <= aired["at"]] or decisions
    inserts = [r for r in reversed(newest_first) if r["event"] == "insert"]
    tel = (row or {}).get("telemetry") or {}
    ev = {"decisions": decisions, "decision": decisions[-1] if decisions else None,
          "aired": aired, "inserts": inserts, "completed": last("completed"),
          "suppressed": last("opportunity_suppressed"), "row": row, "tel": tel}
    ev["delivery"] = _delivery(row, ev["completed"], inserts)
    if ev["suppressed"] and ev["delivery"]["state"] is None:   # it records delivered: False
        ev["delivery"] = {"state": "suppressed", "aired_to_room": False,
                          "reason": ev["suppressed"].get("reason"), "coverage": None,
                          "coverage_source": None, "source": "opportunity_suppressed"}
    # what no free-text field of this break may show, on both routes
    titles = {str(r.get("calendar_event")) for r in recs if r.get("calendar_event")}
    tel_cal = tel.get("calendar_event") if isinstance(tel.get("calendar_event"), dict) else {}
    if tel_cal.get("title"):
        titles.add(str(tel_cal["title"]))
    ev["titles"], ev["announced"] = titles, _carries_announcement(recs, tel, _kind(aired, tel))
    # a row's error is free text (a failed render keeps its exception's message)
    ev["delivery"]["reason"] = _hide(ev["delivery"]["reason"], ev, code_ok=True)
    return ev


def _outcome(ev: dict[str, Any], kind: str) -> str:
    deliv, aired, dec = ev["delivery"], ev["aired"], ev["decision"]
    if ev["suppressed"] or (deliv["state"] == "revoked" and deliv["reason"] in _POLICY_REVOKES):
        return "suppressed"
    if deliv["state"] in ("revoked", "missed", "failed", "insert_failed", "cut"):
        return "delivery_failure"
    if kind != "dj_break":
        return "program"
    if aired is None:
        return "rejected" if dec and dec.get("action") == "say_nothing" else "unknown"
    mode = aired.get("writer_mode")
    if mode in _AS_WRITTEN:
        return "accepted"
    return "fallback" if mode == "deterministic_fallback" else "unknown"


def _writer_outcome(ev: dict[str, Any]) -> str | None:
    aired, dec = ev["aired"], ev["decision"]
    mode = (aired or {}).get("writer_mode")
    if mode in _AS_WRITTEN:
        return "accepted"
    if mode == "deterministic_fallback":
        return "floor"
    if mode:
        return "program"
    if dec:
        return "silent" if dec.get("reason") in _PLANNER_OWN else (
            "refused" if dec.get("action") == "say_nothing" else None)
    return None


def _completion(aired: dict | None) -> tuple[str | None, dict]:
    """How the station finished the copy: the floor wrote it, a station tail
    was appended (required_hosting.completion), or neither."""
    comp = ((aired or {}).get("required_hosting") or {}).get("completion")
    if isinstance(comp, dict) and comp.get("completed"):
        return "station_tail", comp
    if not aired:
        return None, {}
    return ("floor" if aired.get("writer_mode") == "deterministic_fallback" else "none"), {}


def _summary(token: str, recs: list[dict[str, Any]], row: dict | None,
             ev: dict[str, Any] | None = None) -> dict[str, Any]:
    ev = ev or _evidence(recs, row)
    kind = _kind(ev["aired"], ev["tel"])
    outcome = _outcome(ev, kind)
    dec = ev["decision"] or {}
    first = [r["at"] for r in recs] + ([row["created_at"]] if row and not recs else [])
    deliv = ev["delivery"]
    obl = _obligations(ev, deliv)
    queue = next((r.get("queue_id") for r in recs if r.get("queue_id")), None) or (row or {}).get("queue_id")
    out = {
        "opportunity_id": _ref(token), "kind": kind,
        "at": min(first) if first else None, "boundary": dec.get("boundary"), "queue_id": queue,
        "outcome": outcome, "writer_outcome": _writer_outcome(ev),
        "aired_to_room": deliv["aired_to_room"],
        "service_completion_mode": _completion(ev["aired"])[0],
        "delivery": {k: deliv[k] for k in ("state", "reason", "coverage", "source")},
        "tracks": {"ending": dec.get("ending") or ev["tel"].get("intended_pre_name"),
                   "next": dec.get("next") or ev["tel"].get("intended_post_name")},
        "rejection": _why(outcome, ev),
        "obligations_evaluated": (False if ev["suppressed"] else
                                  True if (ev["decision"] or ev["aired"]) else None),
        "obligations": {"owed": obl["owed"], "confirmed_delivered": obl["confirmed_delivered"],
                        "items": [{k: i[k] for k in ("key", "due", "delivered")} for i in obl["items"]]},
        "evidence": {"records": sorted({r["event"] for r in recs}),
                     "break_row": "present" if row else "not_found"},
    }
    return out


def _mask_calendar(text: Any, titles: set[str]) -> Any:
    """Every title, wherever it is said, in one pass, longest first. A title
    is matched word for word, case-insensitively, with any run of whitespace
    between its words: real titles carry double spaces that the floor
    speaks as one. A title is a whole phrase, never part of a
    longer word: "Sam" does not mask the "Sam" in "Samantha".
    Nothing else is loosened."""
    if not isinstance(text, str):
        return text
    words = sorted({tuple(t.split()) for t in titles}, key=lambda w: len(" ".join(w)), reverse=True)
    alts = [r"\s+".join(map(re.escape, w)) for w in words if w]
    return (re.sub(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", "[calendar event]", text, flags=re.I)
            if alts else text)


def _carries_announcement(recs: list[dict[str, Any]], tel: dict, kind: str) -> bool:
    """Did this break speak an announcement's message? Its records never
    name the message (an announcement's `aired` has `calendar_event: null`),
    so it cannot be masked, only recognised: an announcement, a DJ
    break that owed one (a calendar-announcement carrier: `announcement`
    among its owed obligations), or a row that says it carried one."""
    if kind == "announcement":
        return True
    for r in recs:
        obl, hosting = r.get("obligations"), r.get("required_hosting")
        for owed in (r.get("required_obligations"), obl.get("owed") if isinstance(obl, dict) else obl,
                     hosting.get("obligations") if isinstance(hosting, dict) else None):
            if isinstance(owed, list) and "announcement" in owed:
                return True
    return bool(tel.get("announcement") or tel.get("announcement_delivery"))


def _hide(text: Any, ev: dict[str, Any], code_ok: bool = False) -> Any:
    """A free-text field as it may leave: withheld whole if the break carried
    an announcement (a bare reason code survives where `code_ok`), else with
    its calendar titles masked."""
    if not isinstance(text, str) or not text.strip():
        return text
    if ev["announced"] and not (code_ok and _CODE.fullmatch(text)):
        return ANNOUNCEMENT_WITHHELD
    return _mask_calendar(text, ev["titles"])


def _detail(token: str, recs: list[dict[str, Any]], row: dict | None) -> dict[str, Any]:
    ev = _evidence(recs, row)
    out = _summary(token, recs, row, ev)
    mask = lambda s: _hide(s, ev)  # noqa: E731
    aired, dec, deliv = ev["aired"] or {}, ev["decision"] or {}, ev["delivery"]
    writer, wev = aired.get("writer") or {}, aired.get("writer_evidence") or {}
    rej = out["rejection"]
    # a span belongs to the reason shown: the planner's own refusal
    span, span_src = _span(dec if rej["reason_source"] == "planner_decision.reason" else None)
    rej.update(offending_span=mask(span), offending_span_source=span_src,
               fact_ref=({"fact_id": dec.get("fact_id"), "predicate": dec.get("predicate")}
                         if rej["reason_source"] == "planner_decision.reason" and dec.get("fact_id")
                         else None),
               fallback_action=(f"floor {writer.get('template_id')}" if aired.get("writer_mode") ==
                                "deterministic_fallback" and writer.get("template_id") else None))
    if span is None and rej["validation_stage"] in ("planner_guard", "render_writer_guard"):
        rej["offending_span_source"] = "category only; the guard records no span"
    out["smart_dj"] = {
        # deterministic_primary is itself the record that no planner ran: the
        # engine writes it only for an installation with no writing model
        "attempted": True if ev["decisions"] else (
            False if out["kind"] != "dj_break" or aired.get("writer_mode") == "deterministic_primary"
            else None),
        "accepted": aired.get("writer_mode") == "smart_planner" if ev["aired"] else None,
        "provider": writer.get("fallback_from") or (writer.get("provider") if aired.get("writer_mode") ==
                                                     "smart_planner" else None),
        "model": wev.get("model"),
        "attempted_kind": dec.get("attempted") or (dec.get("action") if dec.get("action") != "say_nothing"
                                                   else None),
        "writer_style": dec.get("writer_style"),
        "attempts": len(ev["decisions"]) if ev["decisions"] else None,
        "retry": bool(dec.get("retry")),
        "raw_draft": mask(dec.get("draft") or (dec.get("text") if dec.get("action") != "say_nothing" else None)
                          or wev.get("raw_smart_draft")),
        "draft_word_count": dec.get("draft_word_count") or dec.get("word_count"),
        "fact_ids_cited": list(dec.get("fact_ids_cited") or dec.get("cited_fact_ids") or []),
        "unknown_fact_ids": list(dec.get("unknown_fact_ids") or []),
        "moment_refs": list(dec.get("moment_refs") or []),
        "warnings": [mask(w) for w in dec.get("warnings") or []],
        "decisions": [{"at": d.get("at"), "action": d.get("action"), "reason": d.get("reason"),
                       "guard_category": d.get("guard_category"),
                       "offending_span": mask(_span(d if d.get("action") == "say_nothing" else None)[0]),
                       "draft": mask(d.get("draft"))} for d in ev["decisions"]],
    }
    mode, comp = _completion(ev["aired"])
    out["aired_copy"] = {
        "final_text": mask(aired.get("text")) if ev["aired"] else None,
        "lifecycle": "rendered" if ev["aired"] else None,   # heard only if delivery says so
        "word_count": aired.get("word_count"), "writer_mode": aired.get("writer_mode"),
        "template_id": writer.get("template_id"),
        "floor_aired": (aired.get("writer_mode") == "deterministic_fallback") if ev["aired"] else None,
        "service_completion": mode,
        "service_completion_added": list(comp.get("missing") or []) if comp else None,
        "service_completion_position": comp.get("position") if comp else None,
        "voice": {k: (aired.get("voice") or {}).get(k) for k in ("provider", "voice_id")}
        if aired.get("voice") else None,
    }
    obl = _obligations(ev, deliv)
    wx, cal = aired.get("weather") or dec.get("weather"), aired.get("calendar") or dec.get("calendar")
    for item in obl["items"]:
        if item["key"] == "weather" and isinstance(wx, dict):
            item["context"] = {k: wx.get(k) for k in ("status", "temperature", "condition",
                                                       "observation_age_minutes")}
        if item["key"] == "calendar" and isinstance(cal, dict):
            item["context"] = {k: cal.get(k) for k in ("status", "event_id", "timing_band", "minutes_until")}
    out["obligations"] = obl
    tel, last_insert = ev["tel"], (ev["inserts"][-1] if ev["inserts"] else None)
    render_ready = bool((row or {}).get("duration_s")) or bool(ev["inserts"])
    out["pipeline"] = {
        "writer_state": out["writer_outcome"],
        "writer_detail": (f"{len(ev['decisions'])} planner decision(s); writer mode "
                          f"{aired.get('writer_mode')}") if (ev["decisions"] or ev["aired"]) else None,
        "render_state": "audio_ready" if render_ready else ("copy_accepted" if ev["aired"] else None),
        "render_detail": (f"{round(row['duration_s'], 1)} s audio" if (row or {}).get("duration_s") else None),
        "insert_state": last_insert.get("outcome") if last_insert else None,
        "insert_detail": (f"attempt {last_insert.get('attempt')} at {last_insert.get('stage')}"
                          # free text like the row's error, so the same filter
                          + (f" ({_hide(last_insert['detail'], ev, code_ok=True)})"
                             if last_insert.get("detail") else "")
                          + f"; {len(ev['inserts'])} insert record(s)") if last_insert else None,
        "completion_state": deliv["state"],
        "completion_detail": "; ".join(x for x in (
            f"source {deliv['source']}" if deliv["source"] else None,
            f"coverage {deliv['coverage']} ({deliv['coverage_source']})" if deliv["coverage"] else None,
            ("air start recorded" if tel.get("air_start") else "no air start recorded") if row else None,
            f"row evidence {dj_engine.airing_evidence(row)}" if row else None) if x) or None,
    }
    out["evidence"].update(record_shape=obl["record_shape"],
                           counts={e: sum(r["event"] == e for r in recs) for e in sorted(_EVENTS)},
                           redactions=(["calendar_title"] if ev["titles"] else [])
                           + (["announcement_copy"] if ev["announced"] else []))
    return out


# --- scrubbing and the two queries ---------------------------------------------------

def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        for rx, repl in _SCRUB:
            obj = rx.sub(repl, obj)
        return obj
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _cursor(item: dict[str, Any]) -> str:
    return f"{item['at']}~{item['opportunity_id']}"


def _parse_cursor(before: str | None) -> tuple[str, str] | None:
    if not before:
        return None
    at, _, ref = before.partition("~")
    if not ref or not _REF.match(ref):
        raise ValueError("before must be a cursor returned as next_before")
    datetime.fromisoformat(at)
    return at, ref


def _matches(item: dict[str, Any], outcome: str | None, reason: str | None, kind: str) -> bool:
    if kind != "all" and item["kind"] != kind:
        return False
    if outcome == "no_break_aired":
        if item["aired_to_room"] is not False:
            return False
    elif outcome and item["outcome"] != outcome:
        return False
    # the reason as the list SHOWS it: a row's free-text error is scrubbed on the way out
    return not reason or _scrub(item["rejection"]["reason_code"]) == reason


def list_opportunities(limit: int = 50, before: str | None = None, outcome: str | None = None,
                       reason: str | None = None, kind: str = "dj_break", db_path=None) -> dict[str, Any]:
    """Compact rows, newest first: no drafts, no aired copy, no pipeline."""
    limit = max(1, min(int(limit), LIMIT_MAX))
    cur = _parse_cursor(before)
    if outcome and outcome not in OUTCOMES + ("no_break_aired",):
        raise ValueError(f"outcome must be one of {OUTCOMES + ('no_break_aired',)}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    stats, rows = _stats(), _Rows(db_path)
    try:
        held = rows.tokens()
        found: list[dict[str, Any]] = []

        def consider(token, recs, row):
            item = _summary(token, recs, row)
            if item["at"] and (cur is None or (item["at"], item["opportunity_id"]) < cur) \
                    and _matches(item, outcome, reason, kind):
                found.append(item)
        for recs in _groups(stats):
            if len(found) > limit:
                found.sort(key=lambda i: (i["at"], i["opportunity_id"]), reverse=True)
                del found[limit + 1:]
                if recs[0]["at"] < found[limit]["at"]:
                    break                       # nothing older can enter this page
            token = recs[0]["break_id"]
            consider(token, recs, rows.get(token) if token in held else None)
            held.pop(token, None)
        for token, created in held.items():     # a row the stream never mentions
            if token not in stats["seen"] and (stats["complete"] or (
                    stats["oldest_at"] and created >= stats["oldest_at"])):
                consider(token, [], rows.get(token))
    finally:
        rows.close()
    found.sort(key=lambda i: (i["at"], i["opportunity_id"]), reverse=True)
    more = len(found) > limit or (stats["budget_exhausted"] and bool(found))
    page = found[:limit]
    return _scrub({"ok": True, "schema": SCHEMA, "items": page,
                   "next_before": _cursor(page[-1]) if more and page else None,
                   "scan": {k: stats[k] for k in ("complete", "budget_exhausted", "bytes_read",
                                                  "budget_bytes", "oldest_at", "unreadable_lines",
                                                  "sources")}})


def opportunity_detail(opportunity_id: str, db_path=None) -> dict[str, Any] | None:
    """Everything recorded about one opportunity, or None."""
    if not _REF.match(opportunity_id or ""):
        return None
    stats, rows = _stats(), _Rows(db_path)
    try:
        held = {t for t in rows.tokens() if _ref(t) == opportunity_id}
        for recs in _groups(stats):
            token = recs[0]["break_id"]
            if token in held or _ref(token) == opportunity_id:
                return _scrub({"ok": True, "schema": SCHEMA, **_detail(token, recs, rows.get(token))})
        if held:
            token = next(iter(held))
            return _scrub({"ok": True, "schema": SCHEMA, **_detail(token, [], rows.get(token))})
        return None
    finally:
        rows.close()
