#!/usr/bin/env python3
"""The Newsroom view model: backend state translated into a person's words.

WHY THIS IS A MODULE AND NOT PANEL JAVASCRIPT. The Newsroom shows the
bulletin engine's real state, and that state has a large, precise
vocabulary -- forty-odd lifecycle reasons, a one-slot artifact cache, a
target-hour duplicate guard, an air window measured in seconds. Every one
of those is a fact the backend already knows exactly. Restating them in
the browser would mean a second copy of the vocabulary, drifting, in a
language with no tests against the engine. So the translation lives HERE,
beside the engine, with the engine's own tests around it, and the panel
renders sentences.

WHAT IT MAY NOT DO. It is a view over `news_bulletins.status()` and
nothing else: it opens no socket, fetches no feed, runs no scheduler pass,
writes no state and never raises. A status screen that can fail is a
status screen that lies, and a status screen that can trigger a fetch is
not a status screen at all.

TWO CAPABILITIES, REPORTED SEPARATELY. Scheduled audio bulletins and
headlines inside DJ breaks are different features with different sources.
An audio podcast feed supplies AUDIO; it does not supply reported
headlines, and an episode title is not one. Where there is no textual
headline source, this says so in as many words rather than implying the
bulletin feed covers it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import dj_context
import service_clock

# The two headings the Newsroom is built from.
BULLETINS = "bulletins"
HEADLINES = "headlines_in_breaks"

# A cached bulletin older than this is no longer "the latest news" in any
# useful sense. Mirrors news_schedule.MAX_BULLETIN_AGE_MIN, read from it
# rather than restated, so the number cannot drift between the engine's
# freshness rule and the sentence describing it.


def _max_age_minutes() -> float:
    try:
        import news_schedule
        return float(news_schedule.MAX_BULLETIN_AGE_MIN)
    except Exception:
        return 90.0


# --- lifecycle -> plain English ----------------------------------------------
#
# ONE sentence per state, written for somebody who has never read this
# codebase. The raw enum still travels, under `advanced`, for the places
# that legitimately want it (diagnostics, a bug report, a test).
_STATE_WORDS: dict[str, str] = {
    "disabled": "Off",
    "idle": "Waiting for the next bulletin",
    "watching": "Waiting for the next bulletin",
    "refreshing": "Fetching the latest bulletin",
    "armed": "Ready for the next song boundary",
    "rendering": "Recording the introduction",
    "ready": "Ready to air",
    "queued": "Queued — airs after this song",
    "intro_airing": "On air — introduction",
    "bulletin_airing": "On air — bulletin",
    "completed": "Last bulletin aired",
    "skipped": "This hour was skipped",
    "unavailable": "Status unavailable",
}

# Why a bulletin did not happen, said as something a person could act on.
# Anything not named here degrades to a generic, honest sentence rather
# than leaking the enum.
_REASON_WORDS: dict[str, str] = {
    "bulletins_disabled": "Scheduled bulletins are switched off.",
    "source_unconfigured": "No feed URL has been set.",
    "feed_failed": "The feed could not be read. The music was not "
                   "interrupted.",
    "no_bulletin": "No bulletin has been fetched yet.",
    "bulletin_not_cached": "No bulletin was ready in time. The music "
                           "continued.",
    "bulletin_stale": "The newest item in the feed was too old to air as "
                      "news.",
    "bulletin_published_in_future": "The feed dated its newest item in the "
                                    "future, so it was not aired.",
    "bulletin_published_at_invalid": "The feed's publication date could not "
                                     "be read, so the item was not aired.",
    "bulletin_source_mismatch": "The cached bulletin came from a different "
                                "feed and was not aired.",
    "bulletin_already_aired": "That bulletin had already been on air.",
    "bulletin_no_longer_current": "A newer bulletin replaced this one "
                                  "before it aired.",
    "target_hour_already_completed": "This hour's bulletin has already "
                                     "aired.",
    "target_hour_already_skipped": "This hour was already passed over.",
    "boundary_too_early": "Waiting for a song to finish.",
    "boundary_too_late": "No song ended before this hour's bulletin "
                         "expired, so it was skipped. The music was not "
                         "interrupted.",
    "boundary_late_grace": "The next song ends after the on-time window; "
                           "the bulletin is still fresh and will air "
                           "there.",
    # Another programme already holds the song boundary. Each was blank on
    # the desk before -- the station looked idle while News was blocked.
    "ordinary_break_owns_boundary": "A DJ break already queued holds this "
                                    "song boundary; the bulletin waits for "
                                    "the next one.",
    "announcement_owns_boundary": "An announcement already queued holds "
                                  "this song boundary; the bulletin waits "
                                  "for the next one.",
    "event_liner_owns_boundary": "A listener's request holds this song "
                                 "boundary; the bulletin waits for the "
                                 "next one.",
    "block_separated": "Music Assistant could not keep the introduction and "
                       "the bulletin together, so this hour was skipped. "
                       "The music was not interrupted.",
    "bulletin_retired": "That bulletin can no longer be aired.",
    "boundary_item_changed": "The song that was about to end changed.",
    "before_check_window": "It is not yet time to check the feed.",
    "airtime_unknown": "Waiting until the next song boundary can be "
                       "predicted.",
    "station_clock_unknown": "The station clock is unavailable.",
    "no_active_session": "The station is not playing.",
    "not_playing": "The station is not playing.",
    "queue_changed": "The queue changed before the bulletin could be "
                     "inserted.",
    "queue_session_mismatch": "The queue is being played by something else.",
    "queue_unreadable": "Music Assistant's queue could not be read.",
    "enqueue_failed": "The bulletin could not be handed to Music Assistant. "
                      "The music was not interrupted.",
    "intro_claim_lost": "Another break claimed the same boundary.",
    "intro_render_failed": "The introduction could not be recorded; the "
                           "bulletin was set to air on its own.",
    "disabled_mid_event": "Bulletins were switched off before this one "
                          "aired.",
    "gate_closed": "The boundary closed before the bulletin could be "
                   "inserted.",
    "target_lost": "The scheduled hour was lost and released.",
    "planner_error": "The scheduler hit an unexpected error and stood down. "
                     "The music was not interrupted.",
    "event_in_flight": "A bulletin is already on its way to air.",
    "rendering_intro": "Recording the introduction.",
    "queued": "Queued — airs after this song.",
    "bulletin_fresh": "A fresh bulletin is ready.",
    "boundary_eligible": "A song boundary inside the window is available.",
    "clear": "Nothing is in the way.",
}


# Reasons that mean "working normally, waiting". They are never trouble.
_ORDINARY_REASONS = frozenset((
    "clear", "bulletin_fresh", "queued", "boundary_eligible",
    "boundary_late_grace",
    "rendering_intro", "boundary_too_early", "before_check_window",
    "bulletin_completed", "intro_airing", "bulletin_airing",
))


def _reason_sentence(reason: str | None) -> str | None:
    """A person's sentence for a lifecycle reason, or None.

    `never_aired:<inner>` is the scheduler's compound form for a handoff
    that provably died; `expired:<inner>` for a slot that ran out while
    still waiting. In both the inner reason is the informative half."""
    if not reason:
        return None
    if reason.startswith("expired:"):
        inner = _REASON_WORDS.get(reason.split(":", 1)[1])
        return ("This hour's bulletin ran out of time before it could air. "
                + (inner or "The music was not interrupted."))
    if reason.startswith("never_aired:"):
        inner = _REASON_WORDS.get(reason.split(":", 1)[1])
        return ("The last scheduled bulletin never reached the air and was "
                "released. " + inner) if inner else (
                "The last scheduled bulletin never reached the air and was "
                "released. The music was not interrupted.")
    return _REASON_WORDS.get(reason)


# --- small formatters ---------------------------------------------------------

def _clock(iso: str | None) -> str | None:
    """A stamp as the wall clock a person reads. A naive stamp is the legacy
    station-local encoding and is read as written. An aware one is an
    instant, shown on the STATION's wall clock (`dj_context.station_at`: the
    published zone, else the published offset), because the container's
    zone need not be the home's; only when no station clock was published
    does it fall back to the process zone (Supervisor's, the home's). Ages
    come from `_minutes_since`, which compares instants, so a clock and its
    age agree in every zone."""
    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is not None:
        when = dj_context.station_at(when) or service_clock.to_legacy(when)
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


def _age_words(minutes: float | None) -> str | None:
    """'just now' / '18 minutes ago' / '3 hours ago'. None when unknown."""
    if minutes is None:
        return None
    if minutes < 0:
        return "dated in the future"
    if minutes < 2:
        return "just now"
    if minutes < 90:
        return f"{int(round(minutes))} minutes ago"
    hours = minutes / 60.0
    if hours < 36:
        return f"{int(round(hours))} hours ago"
    return f"{int(round(hours / 24.0))} days ago"


def _duration_words(seconds: float | None) -> str | None:
    if not seconds or seconds <= 0:
        return None
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _minutes_since(iso: str | None, now: datetime | None = None) -> float | None:
    """Minutes from `iso` to `now`. THE one age reading for the Newsroom.

    A naive stamp -- `now` or `iso` -- is the legacy encoding: station-local,
    in the process zone, which under Supervisor is the home's. Relabelling
    it as UTC shifts the age by the UTC offset, so a bulletin aired twenty
    minutes ago would read as hours old anywhere west of UTC: the schedule
    writes `last_aired_at` naive, the feed's `published_at` is aware, and
    only the aware one comes out right."""
    when = service_clock.from_legacy(iso) if iso else None
    if when is None:
        return None
    ref = service_clock.instant(now)
    return (ref - when).total_seconds() / 60.0


def _join(parts: list[str]) -> str:
    """a, b and c -- for a sentence a person reads, not a log line."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def schedule_label(target_minute: int) -> str:
    """How the schedule is offered, in broadcast words."""
    return f"Every hour at :{int(target_minute):02d}"


# --- the two capability blocks ------------------------------------------------

def _bulletins_view(status: dict[str, Any],
                    now: datetime | None = None) -> dict[str, Any]:
    pub = status.get("publisher") or {}
    sched = status.get("schedule") or {}
    current = status.get("current")
    enabled = bool(status.get("enabled"))
    feed_url = (pub.get("feed_url") or "").strip()
    # CONFIGURED means a source the station can fetch. The source list is
    # the Newsroom's model (a legacy single-feed document migrates into it
    # on read); `publisher` is the old single-feed block and stays only for
    # its consumers. A fresh install that adds its feeds in Bulletin
    # Sources leaves `publisher` empty -- which used to read as "No feed
    # configured" over a newsroom that was airing (alpha.4 testing).
    srcs = _sources_view(status, now)
    ready_srcs = [x for x in srcs if x["ready"]]
    configured = bool(ready_srcs) or (bool(feed_url) and not pub.get("error"))
    # The legacy block's own fault is a fault only when it IS the config.
    source_error = None if ready_srcs else pub.get("error")
    minute = sched.get("target_minute")
    minute = 30 if not isinstance(minute, int) or isinstance(minute, bool) else minute
    automation = sched.get("automation") or "unavailable"
    in_flight = automation in ("queued", "intro_airing", "bulletin_airing")

    # HEADLINE: the one thing a person reads first. Configuration outranks
    # the switch, because "On" over an empty feed box would be a promise
    # the station cannot keep.
    if not configured:
        headline = "No feed configured"
        detail = ("Scheduled bulletins need the address of an audio "
                  "podcast or news RSS feed. Add one below and test it.")
    elif not enabled and not in_flight:
        headline = "Bulletins are off"
        detail = ("The feed is set up. Switch bulletins on and the station "
                  "will air one " + schedule_label(minute).lower() + ".")
    else:
        headline = "Bulletins are on"
        detail = ("The station airs a bulletin " + schedule_label(minute).lower()
                  + ", between songs.")

    # LATEST: what is actually in the one-slot cache right now.
    latest = None
    if current:
        age = _minutes_since(current.get("published_at"), now)
        latest = {
            "source": current.get("source_label") or "",
            # The episode's own title. Metadata for a queue row, never a
            # reported headline -- the key says so and so does the UI.
            "item_title": current.get("item_title") or "",
            "published_age": _age_words(age),
            "published_age_minutes": age,
            "duration": _duration_words(current.get("duration_s")),
            "fresh": (age is not None and 0 <= age <= _max_age_minutes()),
        }

    # NEXT WINDOW: the arithmetic the scheduler itself uses, said as a time.
    target_clock = _clock(sched.get("target_at"))
    window = None
    if target_clock:
        start, end = _clock(sched.get("window_start")), _clock(sched.get("window_end"))
        last = _clock(sched.get("expires_at"))
        window = {
            "at": sched.get("target_at"),
            "clock": target_clock,
            "label": (f"Next bulletin around {target_clock}"
                      + (f" (any song ending between {start} and {end}"
                         + (f", or by {last} while it is still fresh"
                            if last else "") + ")"
                         if start and end else "")),
        }

    # TROUBLE: the one sentence worth acting on, chosen from the most
    # specific thing the engine knows. A source error outranks a lifecycle
    # reason, which outranks a stale download error.
    trouble = None
    if source_error and (enabled or feed_url):
        trouble = "This feed cannot be used: " + str(source_error) + "."
    elif enabled and sched.get("reason") not in _ORDINARY_REASONS:
        # Only a reason that reads as a PROBLEM belongs here; the ordinary
        # waiting reasons are already said by the state line, and repeating
        # them as trouble would make a working station look broken.
        trouble = _reason_sentence(sched.get("reason"))
    if trouble is None and status.get("last_error") and enabled and not current:
        trouble = ("The last fetch failed: " + str(status["last_error"])[:160]
                   + ". The music was not interrupted.")

    # WITH MORE THAN ONE SOURCE the single-feed sentences above describe
    # only whichever is next, which reads as though the other is not
    # configured at all. The headline, the schedule chip and the source
    # chip all become plural instead.
    wire = _wire_view(status, srcs, now)
    live_srcs = [x for x in srcs if x["enabled"] and x["ready"]]
    if len(live_srcs) > 1 and enabled:
        minutes = sorted({x["target_minute"] for x in live_srcs
                          if isinstance(x["target_minute"], int)})
        clock = _join([f":{m:02d}" for m in minutes])
        detail = (f"{len(live_srcs)} newsrooms. The station airs a bulletin "
                  f"every hour at {clock}, between songs.")
        schedule_summary = f"Every hour at {clock}"
        source_summary = _join([x["display_name"] or x["name"]
                                for x in live_srcs])
    else:
        one = live_srcs[0] if len(live_srcs) == 1 else None
        schedule_summary = (schedule_label(one["target_minute"])
                            if one and isinstance(one["target_minute"], int)
                            else schedule_label(minute))
        # the live source's own name first: with a source list, a leftover
        # legacy publisher block can name a feed the station no longer uses
        source_summary = ((one and (one["display_name"] or one["name"]))
                          or pub.get("label") or "")

    return {
        "enabled": enabled,
        "configured": configured,
        "source_count": len(live_srcs),
        "headline": headline,
        "detail": detail,
        # TWO names, deliberately. `source_name` is what to SHOW: the
        # owner's word if they gave one, otherwise the feed host. `stored`
        # is what is actually saved, and is the only thing the settings
        # form may seed itself from -- seeding the box with a derived host
        # would turn a fallback into a value the next Save persists.
        "source_name": source_summary,
        "source_name_stored": status.get("source_name") or "",
        "feed_url": feed_url,
        "target_minute": minute,
        "schedule_label": schedule_summary,
        "state_label": _STATE_WORDS.get(automation, "Status unavailable"),
        "on_air": in_flight,
        "latest": latest,
        # ON THE WIRE, one row per configured source. `latest` above is the
        # single newest bulletin the station holds and stays for the
        # single-feed consumers; this is what the Newsroom renders, and it
        # lists a source that has never fetched anything as plainly as one
        # that just did. A configured newsroom is never absent from this
        # list -- absence would read as "not set up", which was exactly the
        # wrong thing to tell an owner whose second feed was configured and
        # simply losing every boundary.
        "wire": wire,
        "next_window": window,
        # THE SOURCE LIST. `source_name` / `feed_url` / `target_minute`
        # above describe whichever source is next, and stay for the
        # single-source consumers; this is what the Newsroom renders.
        # Zero entries is the factory state and not a fault.
        "sources": srcs,
        "last_result": _reason_sentence(sched.get("last_result")),
        "last_result_at": sched.get("last_result_at"),
        "trouble": trouble,
    }


def _sources_view(status: dict[str, Any],
                  now: datetime | None = None) -> list[dict[str, Any]]:
    """Each configured source as sentences a person can act on.

    One row per source, in configured order -- which is also the order two
    sources due for the same minute resolve in, so the screen can say
    plainly which one takes a shared boundary.
    """
    out = []
    for i, src in enumerate(status.get("sources") or []):
        aired_at = src.get("last_aired_at")
        age = _minutes_since(aired_at, now)
        out.append({
            "id": src.get("id") or "",
            "name": src.get("name") or "",
            "display_name": src.get("display_name") or "",
            "feed_url": src.get("feed_url") or "",
            "enabled": bool(src.get("enabled")),
            "target_minute": src.get("target_minute"),
            "schedule_label": schedule_label(src.get("target_minute") or 0),
            # What this source last managed, in its own right. "Never aired"
            # is a real and ordinary answer for a source added ten minutes
            # ago, and is not a fault.
            "last_aired_at": aired_at,
            "last_aired_label": (f"Last aired {_clock(aired_at)}"
                                 + (f" ({_age_words(age)})" if _age_words(age) else "")
                                 if aired_at else "Not aired yet"),
            "last_result": _reason_sentence(src.get("last_result")),
            # THIS source's own newest fetch failure, if it has one. Carried
            # through unchanged so the wire row beside its name can say what
            # went wrong instead of the source simply not appearing.
            "error": src.get("error") or None,
            # A source with no address is a half-finished row, not an error.
            "ready": bool((src.get("feed_url") or "").strip()),
            "order": i,
        })
    return out


def _wire_view(status: dict[str, Any], srcs: list[dict[str, Any]],
               now: datetime | None = None) -> list[dict[str, Any]]:
    """What each configured newsroom currently has, in configured order.

    The row exists because the SOURCE is configured, not because a bulletin
    was fetched. That is the whole correction: the wire used to be built
    from the one-slot cache, so it could only ever show one newsroom -- and
    with the slot global, the one it showed was whichever had refreshed most
    recently. A second configured source was therefore invisible on the
    screen that exists to say what the newsroom is holding, and looked
    unconfigured rather than unlucky.

    A source that is switched OFF is not on the wire at all: the wire is
    what the station is collecting, a disabled source is not being
    collected, and the sources card below it already says so in its own
    row. Five states remain, and each is a different thing to do about it:

      no_feed   a half-finished row with no address yet
      error     the last fetch of THIS source failed, and says how
      waiting   configured and healthy; nothing collected yet this cycle
      stale     holding an episode too old to air
      fresh     holding an episode it can air
    """
    by_source = {}
    for cur in (status.get("currents") or []):
        sid = cur.get("source_id") or ""
        if sid and sid not in by_source:
            by_source[sid] = cur
    max_age = _max_age_minutes()
    out = []
    for src in srcs:
        if not src.get("enabled"):
            continue
        cur = by_source.get(src.get("id") or "")
        age = _minutes_since((cur or {}).get("published_at"), now)
        fresh = (age is not None and 0 <= age <= max_age) if cur else None
        error = src.get("error")
        if not src.get("ready"):
            state, label = "no_feed", "No feed address yet"
        elif cur is None and error:
            state, label = "error", "The last fetch failed"
        elif cur is None:
            state, label = "waiting", "Nothing collected yet"
        elif fresh:
            state, label = "fresh", "Fresh"
        else:
            state, label = "stale", "Too old to air"
        out.append({
            "id": src.get("id") or "",
            "source": src.get("display_name") or src.get("name") or "",
            "schedule_label": src.get("schedule_label") or "",
            "target_minute": src.get("target_minute"),
            "item_title": (cur or {}).get("item_title") or "",
            "published_age": _age_words(age),
            "published_age_minutes": age,
            "duration": _duration_words((cur or {}).get("duration_s")),
            "fresh": fresh,
            "state": state,
            "state_label": label,
            # The source's OWN last failure, never the station's most recent
            # one: a working newsroom must not wear its neighbour's error.
            #
            # AND ONLY WHILE IT IS STILL THE NEWS. A `failed` row survives in
            # the database long after the fetch that wrote it, so a source
            # that failed at noon and succeeded at one would go on displaying
            # the noon error underneath its healthy episode -- telling an
            # owner who had just fixed the thing that it was still broken.
            # The error belongs to the row that has nothing else to say;
            # once the source is holding a bulletin, what it is holding IS
            # the answer.
            "error": (error or None) if cur is None else None,
            "last_aired_label": src.get("last_aired_label") or "",
        })
    return out


def _headlines_view(dj: dict[str, Any] | None) -> dict[str, Any]:
    """Headlines in DJ breaks, reported for what it actually is.

    The engine can weave ONE grounded headline into an ordinary break, but
    only from a textual headline that something published to it with a
    named source. This build ships no such source. Saying anything else --
    implying the bulletin feed covers it, or that switching the break
    toggle on produces headlines -- would be untrue, so the status is
    'not configured' and the explanation says exactly what is missing."""
    dj = dj or {}
    news_cfg = dj.get("news") or {}
    content = dj.get("content") or {}
    sources = [s for s in (news_cfg.get("sources") or []) if str(s).strip()]
    # CONFIGURED means a real textual headline source exists: the feature
    # switched on AND at least one named source the grounding contract can
    # match a published headline against. Neither is inferred from the
    # bulletin feed, which carries audio.
    configured = bool(news_cfg.get("enabled")) and bool(sources)
    return {
        "status": "configured" if configured else "not_configured",
        "label": "Configured" if configured else "Not configured",
        "detail": (
            "The DJ may mention one headline per break, from "
            + ", ".join(str(s) for s in sources) + "."
            if configured else
            "The DJ can mention a news headline in a break, but only from a "
            "written headline with a named source. ParlorWave Radio does "
            "not ship one yet. A podcast episode title is not a headline, "
            "so the bulletin feed above is deliberately not used for this."),
        # The break toggle's own state, reported separately: a person who
        # has it switched on deserves to know it is not producing anything.
        "allowed_in_breaks": bool(content.get("news")),
        "note": (None if configured else
                 "Switching “The DJ may mention a headline” on in the "
                 "Booth changes nothing until a headline source exists."),
    }


# --- the one public call ------------------------------------------------------

def view(status: dict[str, Any] | None = None,
         dj: dict[str, Any] | None = None,
         now: datetime | None = None) -> dict[str, Any]:
    """The whole Newsroom, as sentences, plus the raw state under
    `advanced`.

    Fenced end to end. A view model that throws takes the status screen
    down with it, so anything unexpected degrades to the honest
    'unavailable' shape rather than an exception."""
    try:
        if status is None:
            import news_bulletins
            status = news_bulletins.status()
        if dj is None:
            try:
                import settings
                dj = settings.load().get("dj") or {}
            except Exception:
                dj = {}
        sched = status.get("schedule") or {}
        return {
            BULLETINS: _bulletins_view(status, now),
            HEADLINES: _headlines_view(dj),
            # Raw lifecycle, for diagnostics and bug reports only. The
            # product screen renders the sentences above; these enums are
            # shown only behind Advanced.
            "advanced": {
                "automation": sched.get("automation"),
                "state": sched.get("state"),
                "reason": sched.get("reason"),
                "target_hour": sched.get("target_hour"),
                "target_at": sched.get("target_at"),
                "window_start": sched.get("window_start"),
                "window_end": sched.get("window_end"),
                "last_result": sched.get("last_result"),
                "last_result_at": sched.get("last_result_at"),
                "duplicate_block_reason": sched.get("duplicate_block_reason"),
                "quiet_guard_active": bool(
                    (sched.get("quiet_guard") or {}).get("active")),
                # Deliberately NOT the bulletin token. It is identity AND
                # the credential the unauthenticated audio route accepts,
                # and it tells a person nothing they can act on, so it does
                # not go on a screen. /news/status still carries it for the
                # developer bench.
                "source_id": (status.get("publisher") or {}).get("source_id"),
                "download_in_progress": bool(status.get("download_in_progress")),
                "rows": status.get("rows"),
                "last_error": status.get("last_error"),
            },
        }
    except Exception:  # noqa: BLE001 - a status screen must never be the failure
        return {
            BULLETINS: {"enabled": False, "configured": False,
                        "headline": "Status unavailable",
                        "detail": "The Newsroom could not read the bulletin "
                                  "engine's state. Music is unaffected.",
                        "source_name": "", "source_name_stored": "",
                        "feed_url": "", "target_minute": 30,
                        "schedule_label": schedule_label(30),
                        "state_label": "Status unavailable", "on_air": False,
                        "latest": None, "wire": [], "next_window": None,
                        "last_result": None, "last_result_at": None,
                        "trouble": None},
            HEADLINES: _headlines_view(None),
            "advanced": {"automation": "unavailable"},
        }
