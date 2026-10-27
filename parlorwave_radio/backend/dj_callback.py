"""Deterministic earlier-session callbacks.

The writer is never involved. Handing the model a callback sentence with
the artist masked does not hold: a model can mangle the token or fill the
blank with an artist of its own choosing, airing a false occurrence claim.
The answer is not "mask harder" -- any callback the model can see is a
callback the model can rewrite.

So the callback is built here, from verified session facts, and joined to the
writer's text AFTER the writer is done. This module is pure policy: it reads
no database and calls no model. dj_engine supplies the ledger state and the
recency labeller; everything below is arithmetic over facts.

What a callback may contain: a verified artist name, a verified bounded
recency label, and -- only where a documented relationship supports it --
one clause saying how that artist connects to what is playing. Nothing
else: no title, no mood, no energy, no genre, no comparison, no story.

--- RECENCY IS NOT AN EDITORIAL REASON ----------------------------------

Selecting the most recent artist with an honest recency label and naming
them would make the station open breaks by recalling acts that have
nothing to do with the transition happening, for example:

    "Ada Venn was on a few minutes back. Cold Orchard, Lost Bells ...
     Coming up, Marlow Kites ..."

Ada Venn has no reason to be there. The break has a perfectly good story
-- Cold Orchard out, Marlow Kites in -- and spends its opening sentence on
a third act nobody asked about, which is the same failure mode as
reading a band's roster aloud: a TRUE thing with no reason to be said.

And a RETURN form, recalling an artist who is also the artist coming up,
produces the absurdity it is meant to avoid:

    "we heard Wren Hollis earlier ... and Wren Hollis is up next"

So the rule is about the reason, not the cadence. A callback needs an
EDITORIAL REASON drawn from authoritative structure:

    documented_relationship   the pack documents a link between the
                              recalled artist and the music playing -- a
                              shared member, a collaboration
                              (music_knowledge.artists_related)
    title_reference           the recalled track and the current or next
                              track share a distinctive title phrase

Mere recency is not on that list, and neither is "they are coming up
next" -- that is the writer's handoff to make, in the sentence where it
belongs. No relationship semantics were invented to keep the callback
frequency up: a station that has nothing to recall says nothing, which is
what a real host does.

Every decision, taken or not, leaves a record (`decision`), because "why
was there no callback" is exactly the question the aired copy alone
cannot answer.

--- THE REASON HAS TO BE SAID -------------------------------------------

A callback can have a documented reason -- a member who played in both
bands -- and still never SAY it, for example:

    "Cold Orchard was on earlier tonight. And that one was Marlow
     Kites"

The listener cannot see the relationship table. A callback whose reason
stays in the forensics is, to the room, a callback with no reason -- the
exact failure the rule above exists to end. So the sentence carries the
link, in the pack's own names ("Wren Hollis played in both Cold Orchard
and Marlow Kites"), and a relationship the pack documents but cannot NAME
is not recalled at all (SUPPRESS_UNSPEAKABLE): an unsayable reason is no
reason. A title-reference callback names the recalled track, because
the title IS the reason. Cadence is untouched.
"""

from __future__ import annotations

from typing import Any, Callable

# (minimum minutes since the last aired callback, minimum intervening
# ORDINARY music breaks). One table now: the RETURN class is gone, because
# recalling the artist who is about to play is the one shape that
# reads as absurd on air.
POLICY: dict[str, tuple[int, int]] = {
    "normal": (25, 3),
    "high":   (15, 2),
}
# 'low' is absent on purpose: low chattiness supplies no callback material,
# so no callback can be grounded (the same rule the old callback_depth had).

# A callback break must still be a usable music handoff, so the callback
# spends the writer's words rather than extending the break.
MIN_WRITER_WORDS = 30

# --- the editorial reasons a callback may exist ----------------------------
#: The pack documents a link between the recalled artist and the music
#: playing (music_knowledge.artists_related): a shared member, a
#: collaboration, a founding.
REASON_RELATIONSHIP = "documented_relationship"
#: The recalled track and the current or next track share a distinctive
#: title phrase -- an explicit reference the listener can hear.
REASON_TITLE = "title_reference"
REASONS = (REASON_RELATIONSHIP, REASON_TITLE)

# --- why there was no callback ---------------------------------------------
#: Nothing in the session history had an editorial reason to be recalled.
#: THE ORDINARY ANSWER, and not a defect.
SUPPRESS_NO_RELEVANCE = "no_relevance"
#: The candidate is the artist who just finished: restating the obvious.
SUPPRESS_REDUNDANT_CURRENT = "redundant_with_current"
#: The candidate is the artist coming up next. "We heard Wren Hollis
#: earlier ... and Wren Hollis is up next" -- the destination belongs to the
#: handoff, once.
SUPPRESS_REDUNDANT_NEXT = "redundant_with_next"
#: No honest bounded recency label, so nothing could be said truthfully.
SUPPRESS_TOO_STALE = "too_stale"
#: There is no room: the callback would leave the writer too few words for
#: a usable handoff, and the transition outranks the reminiscence.
SUPPRESS_DISPLACES = "would_displace_current_transition"
#: Nothing has played yet, or an album run is its own continuity.
SUPPRESS_NONE_AVAILABLE = "none_available"
#: The cadence gate (POLICY) says not this break.
SUPPRESS_CADENCE = "cadence"
#: The pack documents a link but cannot name it (no member name, or a
#: relationship type this module has no sentence for). A reason the
#: listener cannot hear is not a reason.
SUPPRESS_UNSPEAKABLE = "relationship_unspeakable"
#: the relationship (or title link) this candidate would state aired
#: inside dj_context.CALLBACK_COOLDOWN_HOURS
SUPPRESS_RECENTLY_TOLD = "relationship_recently_told"

#: The shortest title this module will treat as a reference at all.
#: EXPLICIT means explicit: a shared common noun is not a connection --
#: a five-letter overlap on an everyday word such as "heart" would
#: connect two unrelated titles that merely both contain it, which is a
#: coincidence, not a callback.
TITLE_MIN = 6

_STANDARD_FORMS = (
    "We heard {artist} {recency}.",
    "{artist} was on {recency}.",
    "{artist} came up {recency}.",
)
#: A title reference names the recalled TRACK: the title is the reason.
_TITLE_FORMS = (
    "We heard {artist} with {title} {recency}.",
    "{artist} played {title} {recency}.",
)
#: The link clause, from verified pack facts only: who played in what.
_LINK_SHARED = "{members} played in both {artist} and {other}."
_LINK_MEMBER = "{person} played in {group}."


def _form_index(artist: str, recency: str, n: int) -> int:
    """Stable, state-free rotation: the same facts always phrase the same
    way, and different facts spread across the form set."""
    return sum(ord(c) for c in f"{artist}|{recency}") % n


def eligible(chattiness: str | None, state: dict[str, Any]) -> bool:
    """Is a callback allowed on THIS ordinary music break, by CADENCE?

    Cadence alone never earns a callback -- `select_candidate` still has
    to find an editorial reason -- but a reason on every break would be
    its own tic, so the old air-to-air spacing stands unchanged.

    state carries only what the dj_breaks ledger already knows:
      minutes_since_callback -- None when none has aired this session
      ordinary_since         -- ordinary music breaks since the last callback
      ordinary_total         -- ordinary music breaks so far this session
      session_minutes        -- age of the session
    """
    row = POLICY.get(str(chattiness or "").lower())
    if not row:
        return False                      # low (or unknown) -> disabled
    minutes, gap = row
    since = state.get("minutes_since_callback")
    if since is None:
        # First callback of the session: the same bar, measured against the
        # session itself, so the station does not open by reminiscing.
        return (float(state.get("session_minutes") or 0) >= minutes
                and int(state.get("ordinary_total") or 0) >= gap + 1)
    ordinary_since = int(state.get("ordinary_since") or 0)
    if ordinary_since < 1:
        return False                      # never on consecutive breaks
    return float(since) >= minutes and ordinary_since >= gap


def _title_of(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return (name.split(" - ", 1)[1] if " - " in name else name).strip()


def _norm_title(text: str) -> str:
    import re
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _title_link(recalled_title: str, *others: str) -> bool:
    """Does the recalled track's title EXPLICITLY reference the music
    playing, or the other way round?

    One title has to contain the other, whole and word-bounded -- an
    answer song, a reprise, a part two. A shared word is not a reference:
    half the catalog shares one."""
    mine = _norm_title(recalled_title)
    if len(mine) < TITLE_MIN:
        return False
    for other in others:
        theirs = _norm_title(other)
        if len(theirs) < TITLE_MIN:
            continue
        a, b = (mine, theirs) if len(mine) <= len(theirs) else (theirs, mine)
        if a == b or f" {a} " in f" {b} ":
            return True
    return False


def _join_names(names: list[str]) -> str:
    names = [str(n).strip() for n in names if str(n or "").strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]} and others"


def link_for(relationship: Any, artist: str, other: str
             ) -> dict[str, Any] | None:
    """The SPEAKABLE half of a documented relationship, or None.

    `relationship` is what the oracle returned (music_knowledge
    .artists_related): a `shared_member` record with the members by
    name, or a `direct` 'member of band' record naming the person and
    the group. Anything else -- a bare True, a relationship type with no
    sentence, a shared member the pack never named -- is a link the
    station cannot say, and `sentence` will not guess at one."""
    if not isinstance(relationship, dict):
        return None
    kind = relationship.get("kind")
    if kind == "shared_member":
        members = [m for m in (relationship.get("members") or [])
                   if isinstance(m, str) and m.strip()]
        if not members and relationship.get("detail"):
            members = [str(relationship["detail"])]
        if members and artist and other:
            return {"kind": "shared_member", "members": members[:3],
                    "artist": artist, "other": other}
        return None
    if kind == "direct" and relationship.get("detail") == "member of band":
        person = str(relationship.get("person") or "").strip()
        group = str(relationship.get("group") or "").strip()
        if person and group:
            return {"kind": "member_of", "person": person, "group": group}
    return None


def _clause(link: dict[str, Any] | None) -> str:
    if not link:
        return ""
    if link.get("kind") == "shared_member":
        return _LINK_SHARED.format(members=_join_names(link["members"]),
                                   artist=link["artist"],
                                   other=link["other"])
    if link.get("kind") == "member_of":
        return _LINK_MEMBER.format(person=link["person"],
                                   group=link["group"])
    return ""


def link_key(reason: str | None, artist: str, other: str | None = None,
             title: str | None = None) -> str:
    """The identity of what a callback TELLS, for its cooldown. A
    relationship is the unordered pair of acts (Ada Venn/Cold Orchard is the
    same story told from either side); a title link is the shared title."""
    if reason == REASON_TITLE:
        return "title:" + _norm_title(title or "")
    pair = sorted(_norm_words(x) for x in (artist, other or "") if x)
    return "rel:" + "|".join(pair)


def select_candidate(resolved: list[dict[str, Any]] | None,
                     age_label: Callable[[Any], str | None],
                     pre_name: str | None,
                     post_name: str | None,
                     album_run: bool = False,
                     depth: int = 12,
                     related_fn: Callable[[str, str], Any] | None = None,
                     told: set[str] | frozenset[str] = frozenset()
                     ) -> dict[str, Any]:
    """The decision about recalling an earlier artist, taken or not.

    Returns a DECISION RECORD -- always, never None -- so the caller can
    record why the station did or did not reminisce:

        {"considered": int, "selected": bool, "artist"?, "track"?,
         "recency"?, "reason"?, "suppression_reason"?}

    A candidate must have actually played, carry a truthful bounded
    recency label, be neither the artist who just finished nor the artist
    coming up, AND have a documented editorial reason to be named
    (REASONS above). Recency is not one of them. `related_fn(a, b)` is
    the authoritative relationship lookup (music_knowledge.artists_related
    in production, injected here so this module still reads no database).
    """
    pre_artist = _artist_of(pre_name)
    post_artist = _artist_of(post_name)
    pre_title, post_title = _title_of(pre_name), _title_of(post_name)
    out: dict[str, Any] = {"considered": 0, "selected": False}
    if album_run:
        # an album run is its own continuity
        out["suppression_reason"] = SUPPRESS_NONE_AVAILABLE
        return out
    if pre_artist and pre_artist == post_artist:
        # same artist either side: nothing to recall
        out["suppression_reason"] = SUPPRESS_NONE_AVAILABLE
        return out
    played = [r for r in (resolved or [])
              if isinstance(r, dict) and r.get("outcome") == "played"]
    pool = played[-depth:]
    if not pool:
        out["suppression_reason"] = SUPPRESS_NONE_AVAILABLE
        return out
    seen: set[str] = set()
    why: str | None = None
    for row in reversed(pool):             # newest first
        artist = (row.get("artist") or "").strip()
        if not artist:
            continue
        low = artist.lower()
        if low in seen:
            continue
        seen.add(low)
        if low == pre_artist:
            why = why or SUPPRESS_REDUNDANT_CURRENT
            continue
        if post_artist and low == post_artist:
            # THE UP-NEXT ARTIST CASE. Recalling the act that is about to play
            # duplicates the handoff and reads as a mistake.
            why = SUPPRESS_REDUNDANT_NEXT
            continue
        label = age_label(row.get("resolved_at"))
        if not label:
            why = why or SUPPRESS_TOO_STALE
            continue                       # no honest recency -> no callback
        out["considered"] += 1
        title = (row.get("title") or "").strip()
        reason = None
        link: dict[str, Any] | None = None
        key = None
        if _title_link(title, pre_title, post_title):
            key = link_key(REASON_TITLE, artist, title=title)
            if key in told:
                why = SUPPRESS_RECENTLY_TOLD
                continue
            reason = REASON_TITLE
        elif related_fn is not None:
            for other in (pre_artist and (pre_name or ""),
                          post_artist and (post_name or "")):
                if not other:
                    continue
                rel = related_fn(artist, _artist_name(other))
                if not rel:
                    continue
                # THE REASON HAS TO BE SAYABLE. A documented link
                # the pack cannot name is recorded as such and the
                # station moves on: it does not recall an artist and
                # leave the listener to guess why.
                link = link_for(rel, artist, _artist_name(other))
                if link is None:
                    why = SUPPRESS_UNSPEAKABLE
                    continue
                key = link_key(REASON_RELATIONSHIP, artist, _artist_name(other))
                if key in told:
                    # the same story, told recently: try the next candidate
                    why = SUPPRESS_RECENTLY_TOLD
                    link = None
                    continue
                reason = REASON_RELATIONSHIP
                break
        if reason:
            out.update({"selected": True, "artist": artist,
                        "track": title or None, "recency": label,
                        "reason": reason})
            if link:
                out["link"] = link
            if key:
                out["link_key"] = key
            return out
    out["suppression_reason"] = why or SUPPRESS_NO_RELEVANCE
    return out


def _artist_name(name: str | None) -> str:
    """The artist portion of an 'Artist - Title' name, original casing."""
    if not isinstance(name, str):
        return ""
    return (name.split(" - ", 1)[0] if " - " in name else name).strip()


def _artist_of(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return (name.split(" - ")[0] if " - " in name else name).strip().lower()


def sentence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Render the decision into one occurrence-only line, with the REAL
    artist -- this text never passes through the model.

    OCCURRENCE ONLY, always. The line says the artist was on and how long
    ago, and stops. It no longer claims the destination: announcing what
    is coming up is the writer's job, said once, in the sentence where a
    listener expects it."""
    artist = candidate["artist"]
    recency = candidate["recency"]
    title = str(candidate.get("track") or "").strip()
    if candidate.get("reason") == REASON_TITLE and title:
        # the title IS the reason, so the listener hears it
        idx = _form_index(artist, recency, len(_TITLE_FORMS))
        text = _TITLE_FORMS[idx].format(artist=artist, title=title,
                                        recency=recency)
        form_id = f"title{idx}"
    else:
        idx = _form_index(artist, recency, len(_STANDARD_FORMS))
        text = _STANDARD_FORMS[idx].format(artist=artist, recency=recency)
        form_id = f"standard{idx}"
    # WHY this artist matters to the transition, in verified names. The
    # relationship clause is the whole point of the callback; an
    # occurrence with no reason attached is only a name-check.
    clause = _clause(candidate.get("link"))
    if clause:
        text = f"{text} {clause}"
    out = {"text": text, "artist": artist, "age_label": recency,
           "reason": candidate.get("reason"), "form_id": form_id}
    if candidate.get("link_key"):
        out["link_key"] = candidate["link_key"]
    if candidate.get("link"):
        out["link"] = dict(candidate["link"])
    return out


# --- the destination belongs to ONE voice ----------------------------------
#
# There is no RETURN callback form. It would do two jobs in one
# sentence -- recall that the artist played earlier AND announce they
# are next -- and every failure it can produce is the same failure:
# the destination said twice, once by a module that cannot see the
# writer's text and once by the writer. Invented examples:
#
#   "We heard Ada Venn earlier tonight, and they're back in a minute.
#    That was Cold Orchard with Quarry Light, and here's Ada Venn
#    again."
#   "we heard Wren Hollis earlier ... and Wren Hollis is up next"
#
# Both halves are individually correct and grounded. Together they are
# the station tripping over itself. The callback now only ever states an
# OCCURRENCE, and the artist coming up is never a callback candidate at
# all (SUPPRESS_REDUNDANT_NEXT), so there is nothing left to deduplicate.


def _norm_words(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def writer_budget(base_words: int, callback_text: str | None) -> int | None:
    """The writer's word budget once the callback has taken its share, or
    None when what would be left is too small for a usable handoff (in which
    case the caller drops the callback rather than lengthening the break)."""
    if not callback_text:
        return base_words
    remaining = int(base_words) - len(callback_text.split())
    return remaining if remaining >= MIN_WRITER_WORDS else None


def compose(callback_text: str | None, writer_text: str) -> str:
    """Join the deterministic callback to the accepted writer text."""
    writer_text = (writer_text or "").strip()
    if not callback_text:
        return writer_text
    head = callback_text.strip()
    if head and head[-1] not in ".!?":
        head += "."
    return f"{head} {writer_text}".strip()
