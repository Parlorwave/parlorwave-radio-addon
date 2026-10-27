"""What the station OWES this break, and whether a draft paid it.

Parlorwave has two layers, and only one of them is the model's:

    REQUIRED PROGRAMMING   the previous track, the clock, the next track,
                           weather when weather is due, a calendar event
                           when one is due
    OPTIONAL ENRICHMENT    grounded music knowledge, personality, colour

The writer is responsible for the second. It was never meant to be
authoritative over the first, and without this ledger it quietly was: a
model that answered SAY_NOTHING, or whose draft failed a grounding guard,
took the whole break down with it -- including a weather reading that was
due and a calendar event the listener had asked to be reminded of. A
listener heard silence and had no way to know the station had decided to
speak.

This module is the ledger that separates the two. It states the
obligations for one slot and judges a finished text against them; it
decides nothing about transport, renders nothing, and calls no model.
Every judgement is delegated to the checker that already owns it:

    station_time / weather   service_presence.omitted (the same detectors
                             the airing ledger uses, so a FABRICATED clock
                             is an omission and not a time check)
    calendar                 smart_dj.calendar_spoken (the same anchor test
                             the calendar mark commit uses)
    track anchors            smart_dj._names_in, via `names_music`

Nothing here widens any authority: an obligation exists only when the
value behind it was already resolved and offered by the machinery that
owns it.

THE ASYMMETRY THAT MATTERS. An unmet obligation never relaxes a guard and
never earns a second model call. It only decides WHO hosts this break --
the model's richer draft, or the deterministic floor that is guaranteed to
pay every obligation it can. A rich draft that pays its obligations still
wins; a rich draft that does not loses to a competent host.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("musicbrain")

#: Obligation names. Kept as plain strings because they travel into
#: telemetry and forensics, where a reader needs them to mean something.
ENDING = "ending_track"
NEXT = "next_track"
STATION_TIME = "station_time"
WEATHER = "weather"
CALENDAR = "calendar"
#: Calendar announcements: a plain ANNOUNCE riding this break. Owed
#: whenever the render context carries `announcement_required`; paid only
#: when announce_copy.delivery_problem finds the whole message in the words.
ANNOUNCEMENT = "announcement"

#: Planner SAY_NOTHING reasons that are ENRICHMENT-level: the model had
#: nothing worth adding, or what it added was not true. Neither is a
#: reason for the STATION to stop hosting, so these fall through to the
#: required-hosting floor instead of cancelling the break.
#:
#: Listed exhaustively on purpose. A reason code added later is NOT
#: enrichment-level by default: it keeps the old conservative behaviour
#: (a clean decline) until someone classifies it here deliberately. For a
#: live station that is the safe direction to fail in.
ENRICHMENT_DECLINES = frozenset({
    # the model's own choice, and the guards' verdicts on its draft
    "model_said_nothing",
    "thin_fact_payload", "no_hosting_payload", "no_hosting_anchor",
    "dangling_forward_reference",
    "ungrounded_claim", "ungrounded_text", "guard_rejected",
    "fact_reference_mismatch", "fact_subject_not_said",
    "fact_value_not_said", "fact_relationship_drift",
    # a clock the station did not supply (smart_dj.FAIL_UNSUPPLIED_CLOCK):
    # the floor states the time only when it is owed, and then exactly
    "unsupplied_clock",
    "self_reference", "empty_text", "invalid_result",
    "unknown_kind", "kind_not_eligible",
    # the provider failed to answer at all. The station still owes the
    # slot, and the floor costs no model call, so silence would be a
    # self-inflicted outage rather than editorial restraint.
    "timeout", "provider_error",
    # no song facts reached the packet: listed so the intent is explicit,
    # though `obligations` will find nothing to host and the floor
    # abstains on its own.
    "no_material",
    # the opt-in P2 writer's own refusals (smart_dj.FAIL_TRUNCATED,
    # FAIL_STAGE): a reply cut off at the token cap, or one carrying a
    # spoken stage direction / emote / markdown structure. The copy is
    # refused, never patched, and the floor takes the slot.
    "truncated_generation", "spoken_stage_direction",
    # its content screen (smart_dj.FAIL_CALENDAR, FAIL_MUSIC_CLAIM): the
    # listener's calendar framed as anyone else's, or an unsupported music
    # claim. The floor carries the owed calendar item itself.
    "calendar_misattribution", "unsupported_music_claim",
    # P2 never writes a break that owes the listener's calendar entry
    # (smart_dj.FAIL_P2_CALENDAR): the floor hosts it, in the listener's
    # framing, with every other owed item.
    "p2_calendar_deterministic",
    # the P2 specifics lock (smart_dj.FAIL_SPECIFICS): a name, title, place,
    # year or quantity the break was never given.
    "unsupplied_specifics",
    # Calendar announcements (smart_dj.FAIL_ANNOUNCEMENT*): the draft did
    # not carry the riding message whole, or tagged it on after the next
    # song. The floor reads it verbatim instead.
    "announcement_not_delivered", "announcement_after_outro",
})


def verifiable(name: str | None) -> bool:
    """Can we actually TELL whether a text named this track?

    The identity matcher skips names shorter than
    `smart_dj.ANCHOR_NAME_MIN`, because a two-character title cannot be
    told apart from ordinary words. A name that is entirely unmatchable
    can never be proved spoken, so demanding it would fail every draft
    forever and hand every slot to the floor for a reason no listener
    could hear. Same rule as everywhere else here: what cannot be
    grounded is not asserted.
    """
    import smart_dj
    if not name:
        return False
    import break_packet
    artist, _, title = str(name).partition(" - ")
    return any(len(smart_dj._norm(part)) >= smart_dj.ANCHOR_NAME_MIN
               for part in (artist, break_packet.spoken_title(title)) if part)


def obligations(context: dict[str, Any] | None,
                max_words: int | None = None) -> list[str]:
    """What THIS slot owes the listener, from the render context alone.

    An obligation appears only when the value behind it is already
    resolved and offered: the ending/next names the scheduler captured
    from live queue state, the clock and weather that service_presence
    ruled due for this break, and the one calendar event dj_context found
    inside an unused band. Nothing is inferred here -- if the scheduler
    could not resolve a next track, the station does not owe one.
    """
    import service_presence
    ctx = context or {}
    out: list[str] = []
    if ctx.get("pre_name") and verifiable(ctx.get("pre_name")):
        out.append(ENDING)
    if ctx.get("post_name") and verifiable(ctx.get("post_name")):
        out.append(NEXT)
    try:
        for concept in service_presence.due(ctx, max_words):
            out.append(STATION_TIME if concept == "station_time" else WEATHER)
    except Exception:  # noqa: BLE001 - an obligation ledger never raises
        logger.debug("service presence due() unavailable", exc_info=True)
    if isinstance(ctx.get("calendar_event"), dict) and (
            ctx["calendar_event"].get("title")):
        out.append(CALENDAR)
    if announcement_message(ctx):
        out.append(ANNOUNCEMENT)
    return out


def announcement_message(context: dict[str, Any] | None) -> str | None:
    """The message a riding announcement requires, or None."""
    req = (context or {}).get("announcement_required")
    msg = req.get("message") if isinstance(req, dict) else None
    return str(msg) if isinstance(msg, str) and msg.strip() else None


def announcement_unpaid(text: str, context: dict[str, Any] | None) -> bool:
    msg = announcement_message(context)
    if not msg:
        return False
    import announce_copy
    return announce_copy.delivery_problem(msg, text or "") is not None


def names_music(text: str, name: str | None) -> bool:
    """Did `text` name this track -- by artist or by title? Delegates to
    the planner's own identity matcher so the hosting floor and the
    planner's anchor rule can never disagree about what 'named' means."""
    import break_packet
    import smart_dj
    if not text or not name:
        return False
    artist, _, title = str(name).partition(" - ")
    # The SPOKEN title, for the same reason the writers get it: a host who
    # says "Slow Ferry" has named the track, and holding them to
    # "Slow Ferry (live at the Cinder Rooms, Port Avel - spring
    # 2011)" would mark a correct handoff unpaid.
    packet = {"music": [{"role": "ending", "artist": artist.strip(),
                         "title": break_packet.spoken_title(title)}]}
    return smart_dj._names_in(smart_dj._norm(text), packet)


def unmet(text: str, context: dict[str, Any] | None,
          max_words: int | None = None) -> list[str]:
    """Which of this slot's obligations the finished text did NOT pay.

    THE FULL LEDGER, which is what the deterministic FLOOR is measured
    against -- it pays everything the station resolved. An accepted model
    draft is held to the much looser `draft_shortfall` instead: good
    radio copy is not a checklist.

    Judged against the words the listener will hear, exactly like the
    airing ledger: a break that cites a fact id it never said, or names
    no music at all, has paid nothing. Never raises -- a ledger failure
    must not cost a break, so it reports 'nothing unmet' and lets the
    draft stand.
    """
    owed = obligations(context, max_words)
    if not owed or not (text or "").strip():
        return list(owed)
    ctx = context or {}
    missing: list[str] = []
    try:
        if ENDING in owed and not names_music(text, ctx.get("pre_name")):
            missing.append(ENDING)
        if NEXT in owed and not names_music(text, ctx.get("post_name")):
            missing.append(NEXT)
        if STATION_TIME in owed or WEATHER in owed:
            import service_presence
            for concept in service_presence.omitted(text, ctx, max_words):
                missing.append(STATION_TIME if concept == "station_time"
                               else WEATHER)
        if CALENDAR in owed:
            import smart_dj
            if not smart_dj.calendar_spoken(
                    text, {"calendar": ctx["calendar_event"]}):
                missing.append(CALENDAR)
        if ANNOUNCEMENT in owed and announcement_unpaid(text, ctx):
            missing.append(ANNOUNCEMENT)
    except Exception:  # noqa: BLE001
        logger.warning("required-hosting ledger failed (draft stands)",
                       exc_info=True)
        return []
    return missing


#: Why an ACCEPTED draft did not satisfy the hosted slot.
NO_MUSIC_ANCHOR = "music_anchor"
NO_HOSTING_PAYLOAD = "hosting_payload"
INTERNAL_META = "internal_meta"

#: The station's own bookkeeping, said out loud: a draft such as
#: "Quarter past seven, and the time is due" -- which is the prompt's heading
#: echoed back to the listener. Nothing a host says on air describes a
#: requirement, a schedule of duties or the act of handing over.
_META_RE = re.compile(
    r"\b(?:time|clock|time check|weather|forecast|calendar)\s+"
    r"(?:is|was|'s|are)\s+(?:now\s+)?(?:due|owed|required)\b"
    r"|\brequired\s+(?:service|hosting|programming|reading)\b"
    r"|\bstation\s+service\b"
    r"|\bhand(?:ing)?\s+it\s+(?:back\s+)?over\b")


def internal_meta(text: str) -> bool:
    """Does the copy talk about the station's obligations instead of
    meeting them?"""
    return bool(_META_RE.search((text or "").lower()))


def spoken_service(text: str, context: dict[str, Any] | None) -> set[str]:
    """Which live service readings the text genuinely put on air -- the
    same detectors the airing ledger uses, never a string search."""
    try:
        import dj_context
        heard = set(dj_context.spoken_service_concepts(text, context or {}))
    except Exception:  # noqa: BLE001
        return set()
    ctx = context or {}
    try:
        import smart_dj
        if isinstance(ctx.get("calendar_event"), dict) and \
                smart_dj.calendar_spoken(text, {"calendar": ctx["calendar_event"]}):
            heard.add(CALENDAR)
    except Exception:  # noqa: BLE001
        pass
    return heard


def draft_shortfall(text: str, context: dict[str, Any] | None,
                    max_words: int | None = None) -> list[str]:
    """Why this ACCEPTED draft does not get to satisfy the hosted slot.

    A DIFFERENT and much looser question than `unmet`. `unmet` asks what
    the floor should pay; this asks only whether the model's own copy is
    real radio. Smart DJ is allowed to be naturally good without
    following a script, so:

      * the OUTGOING track is NOT required. "It has cooled off a
        little, 60 degrees now, so here is a change of scene, from
        Wren Hollis" is valid hosting: grounded, forward,
        and it never names the song that just ended.
      * ONE music anchor is enough -- previous OR next.

    (The CLOCK used to be optional here too. That let Smart breaks air
    in slots that owed the time -- with no time at all, with only "this
    morning", or with an approximation. A due
    clock is a station duty like weather, so it is now judged below.)

    What still fails, because it is not hosting:
      * no music anchor at all ("It's mild and cloudy.")
      * a trivia fragment with no hosting payload: it names the outgoing
        artist but points nowhere and carries no live reading ("Ada
        Venn played in The Lantern Quarter."). A break that names what
        is NEXT, or carries a weather/clock/calendar reading, is
        oriented in the moment; one that does neither is a fact read
        aloud over a transition it never mentions.
      * a DUE service obligation the copy skipped -- the clock, weather
        or a calendar event. Those are the station's duties, not the
        model's taste. A due clock is paid only by the supplied time said
        exactly once and unhedged (`dj_context.spoken_clock`, the same
        rule the ledger records `clock_aired` by): "this morning", "close
        to 7:15 AM" or a second time check is not a time check.
      * copy that narrates the station's bookkeeping ("the time is due",
        "required service", "I'll hand it over") instead of hosting.

    Never raises: a ledger failure lets the draft stand.
    """
    ctx = context or {}
    if not (text or "").strip():
        return [NO_MUSIC_ANCHOR]
    try:
        owed = obligations(ctx, max_words)
        out: list[str] = []
        has_ending = ENDING in owed and names_music(text, ctx.get("pre_name"))
        has_next = NEXT in owed and names_music(text, ctx.get("post_name"))
        track_context = ENDING in owed or NEXT in owed
        if track_context and not (has_ending or has_next):
            out.append(NO_MUSIC_ANCHOR)
        heard = spoken_service(text, ctx)
        situational = bool(heard) or has_next
        if track_context and (has_ending or has_next) and not situational:
            # named the outgoing song and nothing else: a fact fragment
            out.append(NO_HOSTING_PAYLOAD)
        # the station's own duties, which the model does not get to skip
        if STATION_TIME in owed:
            import dj_context
            if dj_context.spoken_clock(text, ctx) is None:
                out.append(STATION_TIME)
        if WEATHER in owed and "weather" not in heard:
            out.append(WEATHER)
        if CALENDAR in owed and CALENDAR not in heard:
            out.append(CALENDAR)
        # the operator's message is never optional: the whole of it, or the
        # floor delivers it verbatim
        if ANNOUNCEMENT in owed and announcement_unpaid(text, ctx):
            out.append(ANNOUNCEMENT)
        if internal_meta(text):
            out.append(INTERNAL_META)
        return out
    except Exception:  # noqa: BLE001
        logger.warning("required-hosting draft check failed (draft stands)",
                       exc_info=True)
        return []


def hostable(context: dict[str, Any] | None) -> bool:
    """Is there authoritative material for a truthful minimum host?

    One track identity is the floor: the deterministic writer can say
    'That was X' or 'Up next, Y', and either is legitimate radio. With
    neither there is nothing truthful to say and the station stays quiet
    -- which is the ONE silence this design preserves on the
    enrichment-decline path.
    """
    ctx = context or {}
    return bool(ctx.get("pre_name") or ctx.get("post_name")
                or announcement_message(ctx))


# --- complete, don't replace -------------------------------------------------
#: The shortfalls a Smart draft may be COMPLETED for. Both are station
#: readings the deterministic floor would state in fixed words from the same
#: supplied values; everything else (no music anchor, no payload, a skipped
#: calendar event, bookkeeping) still hands the slot to the floor.
COMPLETABLE = frozenset({STATION_TIME, WEATHER})


def _completion_clause(missing: list[str], ctx: dict[str, Any],
                       draft: str = "") -> str | None:
    """The floor's own wording for what is missing, from supplied values
    only: deterministic_writer's CLOCK ('clock.its' / 'clock.the_time_is')
    and WEATHER ('wx.outside' / 'wx.currently') shapes, joined as the
    floor's combined templates join them. When the draft already says
    "it's", the alternate shapes keep two sentences from opening alike."""
    import deterministic_writer
    said_its = bool(re.search(r"\bit[\u2019']s\b", draft or "", re.I))
    clock = str(ctx.get("station_time") or "").strip()
    wx = deterministic_writer._weather_clause(
        str(ctx.get("current_weather") or ""))
    need_clock, need_wx = STATION_TIME in missing, WEATHER in missing
    if (need_clock and not clock) or (need_wx and not wx):
        return None
    if need_clock and need_wx:
        return f"It's {clock}, and it's {wx} outside."
    if need_clock:
        return f"The time is {clock}." if said_its else f"It's {clock}."
    return (f"Currently {wx}." if said_its else f"It's {wx} outside.")


def complete_service(text: str, missing: list[str],
                     context: dict[str, Any] | None,
                     max_words: int | None = None
                     ) -> tuple[str | None, dict[str, Any]]:
    """An otherwise valid Smart draft that skipped ONLY a due clock and/or
    due weather, completed with the floor's authoritative wording -- or
    (None, why) when it cannot be done safely.

    WHY. A shortfall used to hand the whole slot to the floor, and that
    threw away good drafts (a career-span line such as "Cold Orchard
    formed in '86, and Gullwing arrived nineteen years on"; a
    shared release year; a producer credit) to air a
    template that dropped their facts. The obligation is the station's, so
    the station can pay it in its own words without discarding the host's.

    Safety, in order:
      * only COMPLETABLE shortfalls, and only from supplied values;
      * the draft must state NO clock (when the clock is missing) and NO
        temperature reading (when the weather is): a second, different
        reading beside the authoritative one is a contradiction, so that
        draft still goes to the floor;
      * ORDER: the clause goes before the closing handoff when the last
        sentence names the next track ("...It's 8:40 PM. Here's Cold
        Orchard."), otherwise at the end;
      * the whole stays inside the word budget;
      * and the completed text must have NO shortfall at all, judged by the
        same `draft_shortfall` that refused the draft. The caller re-runs
        the authority guard on it as well.
    Never raises."""
    ctx = context or {}
    info: dict[str, Any] = {"missing": list(missing or [])}
    try:
        import smart_dj
        if not missing or not set(missing) <= COMPLETABLE:
            return None, {**info, "refused": "not_completable"}
        clause = _completion_clause(list(missing), ctx, text)
        if not clause:
            return None, {**info, "refused": "no_supplied_value"}
        if STATION_TIME in missing:
            # a clock in the draft might be a wrong time check; the riding
            # announcement's own times are not (they are the message)
            own = set(smart_dj.clock_tokens(announcement_message(ctx) or ""))
            if [c for c in smart_dj.clock_tokens(text) if c not in own]:
                return None, {**info, "refused": "draft_states_a_clock"}
        if WEATHER in missing and smart_dj._RECENT_TEMP_RE.search(text):
            return None, {**info, "refused": "draft_states_a_reading"}
        sentences = [s for s in smart_dj._SENTENCE_SPLIT_RE.split(
            (text or "").strip()) if s]
        position = "end"
        if (len(sentences) > 1
                and names_music(sentences[-1], ctx.get("post_name"))):
            sentences.insert(len(sentences) - 1, clause)
            position = "before_handoff"
        else:
            last = sentences[-1] if sentences else ""
            if last and last[-1] not in ".!?…'\"’”":
                sentences[-1] = last + "."
            sentences.append(clause)
        out = " ".join(sentences)
        if max_words and len(out.split()) > int(max_words):
            return None, {**info, "refused": "over_budget"}
        still = draft_shortfall(out, ctx, max_words)
        if still:
            return None, {**info, "refused": "still_short", "still": still}
        return out, {**info, "clause": clause, "position": position}
    except Exception:  # noqa: BLE001
        logger.warning("service completion failed (floor takes the slot)",
                       exc_info=True)
        return None, {**info, "refused": "error"}
