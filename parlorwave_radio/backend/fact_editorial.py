"""Which TRUE, RELEVANT facts are worth AIRTIME -- and the relationships
between them that are.

The relevance layer (fact_relevance) answers "what does this fact have to
do with this transition?". A station can have that under control and
still sound worse than its deterministic floor, and the complaint is not
prose. It is:

    the facts are boring, unrelated, or have no point.

"Cold Orchard are an American band." "Wren Hollis -- Northern
Static arrived in 2009, on a major label." "Ada Venn once took
home a rock award." Every one true, scope-safe, about the right
music, and none of them radio. So this module sits ABOVE the raw
knowledge store and below the writer and decides two things the
relevance gate never could:

  SPEAKABILITY   whether a KIND of fact, with THIS value, is worth saying
                 on its own. A record label exists for every album and
                 says nothing; a country of origin is true of most of
                 the pack's artists; a release year read out
                 with no observation attached is a database talking.

  BRIDGES        whether two facts, one per side of the transition (or
                 two about one act), form a RELATIONSHIP that is itself
                 the thing worth saying. "Gullwing came out in 1991" and
                 "Brass Weather came out in 1991" are two dull rows; "both of
                 these landed in '91" is a line. The writer is handed the
                 relationship, structured and evidenced, never two rows
                 and a hope that it notices.

RAW METADATA IS NOT A DJ-WORTHY FACT. Every fact keeps one of three
classes, decided by predicate semantics and the value, never by a
floating-point score:

    reference_only     grounding and bridge evidence; never spoken alone
    bridge_candidate   worth saying only as one half of a relationship
    dj_ready           interesting enough to be spoken on its own

What reaches the writer is: at most a couple of bridges, at most a couple
of dj_ready facts, and very often nothing -- which is a GOOD state. The
deterministic floor is already good radio; optional enrichment has to
beat it, not fill it.

Every bridge carries its type, its subjects, the pack fact_ids that are
its evidence, its values, and the realization policy the validator holds
the writer to. Nothing here consults a model, reads a database, invents a
relationship, or promotes a weak fact to keep the packet full. Pure and
deterministic: the same pools and the same date give the same offer.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime
from typing import Any

import fact_relevance

# --- classes ----------------------------------------------------------------------
REFERENCE_ONLY = "reference_only"
BRIDGE_CANDIDATE = "bridge_candidate"
DJ_READY = "dj_ready"
CLASSES = (REFERENCE_ONLY, BRIDGE_CANDIDATE, DJ_READY)

# --- why a fact is not offered on its own ----------------------------------------
#: A label row exists for every album; it is provenance, not a point.
LABEL_IS_REFERENCE = "label_is_reference"
#: "From the United States" is true of most of the pack.
GENERIC_COUNTRY = "generic_country"
#: A year with no observation attached is a database reading itself out.
BARE_DATE = "bare_date"
#: The pack records an award's NAME and nothing else -- no work, no
#: year. An artist-scoped award (a breakthrough-act award from the act's
#: first years) can come out as "Ada Venn's work, notably Northern
#: Static, secured her that award" about a record made long after -- the
#: credit drifts onto whatever record is in the
#: packet, and no validator can see that. Until a pack carries the year
#: or the work, every award is reference material. (Hall of Fame
#: inductions are about the act as a whole and stay speakable.)
AWARD_CONTEXT_UNRESOLVED = "award_context_unresolved"
#: The band produced its own record: a credit, not a story.
SELF_PRODUCED = "self_produced"
#: Ordinary running time (the retriever decides `notable`).
ORDINARY_LENGTH = "ordinary_length"
#: Personnel keep the existing rule: only a documented link earns them.
PERSONNEL_NEEDS_LINK = "personnel_needs_link"
#: A pressing date, a birthday with no occasion, a memorial date.
PROVENANCE_ONLY = "provenance_only"
#: Whitelisted by the retriever, but nobody has decided how it may be
#: spoken alone. Conservative by construction.
UNCLASSIFIED = "unclassified_predicate"
#: The fact's material already travels inside an offered bridge.
FOLDED_INTO_BRIDGE = "folded_into_bridge"
#: Eligible, but the offer was full of better material.
OVER_BUDGET = "over_budget"

#: How many of each may reach the writer. Fewer, better facts.
MAX_BRIDGES = 2
MAX_DJ_READY = 2
MAX_OFFER = 3
#: Withheld date facts that travel to the VALIDATOR only, as grounding
#: for a correct mention the writer makes on its own (never to the writer).
MAX_REFERENCE = 4

# --- bridge types -----------------------------------------------------------------
SAME_RELEASE_YEAR = "same_release_year"
RELEASE_YEAR_GAP = "release_year_gap"
CAREER_SPAN = "career_span"
SAME_PRODUCER = "same_producer"
SHARED_MEMBER = "shared_member"
SAME_ORIGIN = "same_origin"
SAME_FORMED_YEAR = "same_formed_year"
SAME_RECORDING_STUDIO = "same_recording_studio"
SAME_RECORDING_CITY = "same_recording_city"
RELEASE_ANNIVERSARY = "release_anniversary"
BIRTHDAY = "birthday"
RECENT_RELEASE = "recent_release"
#: Derived so the forensics can show them, and deliberately NOT offered.
SAME_LABEL = "same_label"
SAME_DECADE = "same_decade"

BRIDGE_TYPES = (SHARED_MEMBER, SAME_PRODUCER, SAME_RECORDING_STUDIO,
                RELEASE_ANNIVERSARY, BIRTHDAY, RELEASE_YEAR_GAP, CAREER_SPAN,
                SAME_RELEASE_YEAR, SAME_RECORDING_CITY, SAME_ORIGIN,
                SAME_FORMED_YEAR, RECENT_RELEASE, SAME_LABEL, SAME_DECADE)

#: Why a derivable relationship is withheld.
LABEL_BRIDGES_DISABLED = "label_bridges_disabled"
SAME_DECADE_NOT_A_POINT = "same_decade_is_not_a_point"
GAP_TOO_SMALL = "gap_below_threshold"
SPAN_TOO_SHORT = "span_below_threshold"
SAME_ACT = "same_act_both_sides"

#: A release gap has to be a JUMP to be worth remarking on: two decades.
MIN_YEAR_GAP = 20
#: A band still releasing this long after forming is a story.
MIN_CAREER_SPAN = 20
#: A release inside this many calendar years is news, not history.
RECENT_RELEASE_YEARS = 1

#: Strongest first. Deterministic order for the offer; a type absent
#: here is never offered.
_RANK = {
    SHARED_MEMBER: 1.00, SAME_PRODUCER: 0.95, SAME_RECORDING_STUDIO: 0.95,
    RELEASE_ANNIVERSARY: 0.94, BIRTHDAY: 0.93, RECENT_RELEASE: 0.90,
    CAREER_SPAN: 0.86, RELEASE_YEAR_GAP: 0.85, SAME_RELEASE_YEAR: 0.80,
    SAME_RECORDING_CITY: 0.80, SAME_ORIGIN: 0.75, SAME_FORMED_YEAR: 0.70,
}

#: Among dj_ready facts, which KIND leads when several qualify. A Hall of
#: Fame induction or a named producer is a stronger line than a
#: breakthrough award; retrieval rank breaks ties.
_DJ_READY_RANK = {
    "hall_of_fame_induction": 1.0, "album_producer": 0.9,
    "band_member": 0.9, "member_of_group": 0.9,
    "cross_band_membership": 0.9, "recording_studio": 0.9,
    "recording_location": 0.85, "recording_city": 0.85,
    "group_origin_location": 0.8, "artist_origin_location": 0.8,
    "recording_length": 0.5,
}

#: Words a spoken bridge must contain to count as having stated the
#: RELATIONSHIP (one is enough), mirroring music_knowledge._REALIZE_REQUIRE.
_RELEASE_WORDS = ("releas", "came out", "come out", "arriv", "appear",
                  "emerg", "drop", "issued", "out in", "back in",
                  "dates from", "landed", "from")
#: A "both sides" bridge has to SAY the connection, not merely name the
#: two subjects and the value: "Gullwing came out in 1991. Cold Orchard is
#: next." names both acts and the year and connects nothing. So the
#: required words are the JOINING words, never the release verbs a
#: single fact would satisfy.
_JOIN_WORDS = ("both", "same year", "too", "as well", "also", "alike",
               "so is", "so was", "so did", "each", "the same", "likewise",
               "share", "shared", "in common", "twin")
_REQUIRE = {
    SAME_RELEASE_YEAR: _JOIN_WORDS,
    # the JOIN words for a span: "nineteen years" alone satisfies
    # nothing, because "had gone quiet nineteen years before" (a
    # plausible draft) inverts the relationship while containing the number
    RELEASE_YEAR_GAP: ("apart", "gap", "between", "later", "earlier",
                       "after", "before", "jump", "span", "on from",
                       "separate", "sit"),
    CAREER_SPAN: ("later", "since", "after", "on from", "span", "into",
                  "still", "gap", "between", "on since", "years on"),
    SAME_PRODUCER: _JOIN_WORDS + ("same producer", "same hands",
                                  "same desk"),
    SHARED_MEMBER: _JOIN_WORDS + ("before", "after", "went on", "later",
                                  "moved", "left", "joined", "then",
                                  "between", "connect", "link"),
    SAME_ORIGIN: _JOIN_WORDS + ("same town", "same city", "same part",
                                "same place", "like"),
    SAME_FORMED_YEAR: _JOIN_WORDS + ("same class", "same vintage"),
    SAME_RECORDING_STUDIO: _JOIN_WORDS + ("same studio", "same room"),
    SAME_RECORDING_CITY: _JOIN_WORDS + ("same city", "same town"),
    RELEASE_ANNIVERSARY: ("today", "ago", "anniversary", "this day",
                          "this date"),
    BIRTHDAY: ("today", "turns", "turned", "this very day", "celebrat",
               "this morning", "this afternoon", "this evening", "tonight",
               "milestone day"),
    RECENT_RELEASE: _RELEASE_WORDS + ("new", "latest", "this year",
                                      "last year"),
}

_PERSONNEL = fact_relevance.PERSONNEL
_ORIGIN = ("group_origin_location", "artist_origin_location")
_RECORDING_PLACE = {SAME_RECORDING_STUDIO: ("recording_studio",),
                    SAME_RECORDING_CITY: ("recording_location",
                                          "recording_city")}

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def number_words(n: int) -> str:
    """0..99 as spoken words ('thirty-one'); larger numbers as digits."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if 0 <= n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    return str(n)


def span_words(n: int) -> list[str]:
    """Every way a host may state a span of `n` years so it counts as
    said: the digits, the words, and -- from two decades up -- the
    rounded decade count ("two decades", "nearly four decades")."""
    out = [str(n), number_words(n)]
    if n >= 20:
        decades = int(round(n / 10.0))
        out.append(f"{number_words(decades)} decades")
    return out


def _norm(text: Any) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _same(a: Any, b: Any) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na) and na == nb


def _contains(a: Any, b: Any) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na and nb) and (na in nb or nb in na)


def _year(value: Any) -> int | None:
    m = re.match(r"\s*((?:19|20)\d\d)", str(value or ""))
    return int(m.group(1)) if m else None


def _full_date(value: Any) -> date | None:
    try:
        return datetime.strptime(str(value or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


_STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "at", "to",
         "by", "with", "record", "records"}


def _tokens(value: Any) -> list[str]:
    words = [w for w in re.split(r"[^\w']+", str(value or "").lower())
             if len(w) >= 3 and w not in _STOP]
    return words or [str(value or "").lower().strip()]


def _country_level(fact: dict[str, Any]) -> bool:
    """Is this origin a whole country? The retriever states it (`bland`,
    from its own country table); a fact that does not say is treated as
    generic, because a wrong "this is a real place" costs a dull line on
    air and a wrong "this is a country" costs nothing."""
    return bool(fact.get("bland", True))


def _place(fact: dict[str, Any]) -> str:
    return str(fact.get("place") or fact.get("value") or "")


# --- speakability -----------------------------------------------------------------

def classify(fact: dict[str, Any],
             member_names: set[str] | None = None) -> tuple[str, str | None]:
    """(class, reason-when-not-dj_ready). Predicate semantics first, the
    value second, never a score. Unknown predicates are reference_only:
    nobody has decided how they may be spoken alone. `member_names` are
    the people the pack lists in the acts at this break, so a record a
    member produced counts as self-produced."""
    if not isinstance(fact, dict):
        return REFERENCE_ONLY, UNCLASSIFIED
    pred = str(fact.get("predicate") or "")
    value = str(fact.get("value") or "")
    if pred == "album_record_label":
        return REFERENCE_ONLY, LABEL_IS_REFERENCE
    if pred == "original_recording_release_date":
        return BRIDGE_CANDIDATE, BARE_DATE
    if pred == "group_formed_year":
        return BRIDGE_CANDIDATE, BARE_DATE
    if pred == "date_of_birth":
        return BRIDGE_CANDIDATE, PROVENANCE_ONLY
    if pred in ("group_disbanded_year",):
        return REFERENCE_ONLY, BARE_DATE
    if pred in ("reissue_release_date", "date_of_death"):
        return REFERENCE_ONLY, PROVENANCE_ONLY
    if pred == "recording_length":
        return (DJ_READY, None) if fact.get("notable") else (
            REFERENCE_ONLY, ORDINARY_LENGTH)
    if pred in _ORIGIN:
        if _country_level(fact):
            return REFERENCE_ONLY, GENERIC_COUNTRY
        return DJ_READY, None
    if pred in _PERSONNEL:
        if fact.get("relation") in fact_relevance.STRONG:
            return DJ_READY, None
        return REFERENCE_ONLY, PERSONNEL_NEEDS_LINK
    if pred in ("grammy_award", "prestigious_award"):
        return REFERENCE_ONLY, AWARD_CONTEXT_UNRESOLVED
    if pred == "hall_of_fame_induction":
        return DJ_READY, None
    if pred == "album_producer":
        artist = str(fact.get("artist") or "")
        if artist and _contains(artist, value):
            return REFERENCE_ONLY, SELF_PRODUCED
        if any(_same(value, m) for m in (member_names or ()) if m):
            return REFERENCE_ONLY, SELF_PRODUCED
        return DJ_READY, None
    if pred in ("recording_studio", "recording_location", "recording_city"):
        return DJ_READY, None
    return REFERENCE_ONLY, UNCLASSIFIED


# --- bridges ----------------------------------------------------------------------

def _short(fact_id: str) -> str:
    fid = str(fact_id or "")
    return fid if fid.startswith(("kb:", "br:")) else "kb:" + fid[:12]


def _bridge_id(kind: str, *evidence: str) -> str:
    h = hashlib.sha256(("|".join([kind, *sorted(evidence)])).encode(
        "utf-8")).hexdigest()
    return "br:" + h[:12]


def _side_names(side: dict[str, Any]) -> tuple[str, str]:
    return (str(side.get("artist") or "").strip(),
            str(side.get("album") or "").strip())


def _album_phrase(album: str, artist: str) -> str:
    """How the album is SAID: 'Gullwing by Cold Orchard', or for a
    self-titled record 'Wren Hollis' self-titled album' rather than 'Wren
    Hollis by Wren Hollis'."""
    if _same(album, artist):
        return f"{artist}'{'' if artist.endswith('s') else 's'} self-titled album"
    return f"{album} by {artist}"


def _subject_alts(*names: str) -> list[str]:
    return [n for n in names if n and len(_norm(n)) >= 3]


def _one(pool: list[dict[str, Any]], predicate: str
         ) -> dict[str, Any] | None:
    for f in pool:
        if f.get("predicate") == predicate:
            return f
    return None


def _all(pool: list[dict[str, Any]], predicate: str) -> list[dict[str, Any]]:
    return [f for f in pool if f.get("predicate") == predicate]


def _meta(fact: dict[str, Any]) -> dict[str, Any]:
    meta = fact.get("meta")
    return meta if isinstance(meta, dict) else {}


def _bridge(kind: str, about: str, evidence: list[dict[str, Any]], *,
            text: str, subjects: list[list[str]], values: dict[str, Any],
            objects: list[str], grounds: list[str], eligible: bool = True,
            reason: str | None = None) -> dict[str, Any]:
    fids = [str(f.get("fact_id") or f.get("id") or "") for f in evidence]
    bid = _bridge_id(kind, *fids)
    grounds_all = list(grounds)
    for f in evidence:
        for g in f.get("grounds") or []:
            if g and g not in grounds_all:
                grounds_all.append(str(g))
    subjects = [[s for s in group if s] for group in subjects]
    subjects = [g for g in subjects if g]
    return {
        "id": bid, "fact_id": bid, "kind": "bridge", "bridge_type": kind,
        "about": about, "subject_type": "bridge",
        "predicate": f"bridge:{kind}", "scope": "bridge",
        "value": str(values.get("spoken") or next(iter(values.values()), "")),
        "values": {k: v for k, v in values.items() if k != "spoken"},
        "text": text,
        # no single anchor: the validator holds a bridge to EVERY subject
        "anchor": "",
        "subjects": subjects,
        "grounds": [g for g in grounds_all if g],
        "realize": {"predicate": f"bridge:{kind}",
                    "require": list(_REQUIRE.get(kind, ())),
                    "object": [o for o in objects if o],
                    "subjects": subjects},
        "evidence": [f for f in fids if f],
        "relation": f"bridge:{kind}",
        "score": _RANK.get(kind, 0.0),
        "eligible": bool(eligible),
        "reason": reason,
    }


def _release(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The release date a date RELATIONSHIP may be built from: a studio
    album's only. Retrieval marks each album fact with `release_kind`; one
    marked anything else (or unknown) never seeds a span, a gap, a shared
    year or an anniversary -- a "self-titled album ... N years later"
    line built on a much later compilation's date is the failure
    (music_knowledge _resolve_album).
    A fact with no marking at all is a caller that predates
    the marking and is trusted as before."""
    rel = _one(pool, "original_recording_release_date")
    if rel is not None and "release_kind" in rel and \
            rel.get("release_kind") != "studio":
        return None
    return rel


def _pair_bridges(a: list[dict[str, Any]], b: list[dict[str, Any]],
                  sa: dict[str, Any], sb: dict[str, Any],
                  same_act: bool) -> list[dict[str, Any]]:
    """Relationships BETWEEN the two sides of the transition."""
    out: list[dict[str, Any]] = []
    a_artist, a_album = _side_names(sa)
    b_artist, b_album = _side_names(sb)
    subj_a = _subject_alts(a_album, a_artist)
    subj_b = _subject_alts(b_album, b_artist)
    art_a = _subject_alts(a_artist)
    art_b = _subject_alts(b_artist)
    # release years ---------------------------------------------------------
    ra, rb = _release(a), _release(b)
    if ra and rb and ra.get("fact_id") != rb.get("fact_id") and \
            not _same(a_album, b_album):
        ya, yb = _year(ra.get("value")), _year(rb.get("value"))
        if ya and yb:
            gap = abs(ya - yb)
            if gap == 0:
                out.append(_bridge(
                    SAME_RELEASE_YEAR, "both", [ra, rb],
                    text=(f"{_album_phrase(a_album, a_artist)} and "
                          f"{_album_phrase(b_album, b_artist)} both came "
                          f"out in {ya}."),
                    subjects=[subj_a, subj_b],
                    values={"year": str(ya)}, objects=[str(ya)],
                    grounds=[str(ya), "same year", "both came out",
                             "came out", "released"]))
            elif gap >= MIN_YEAR_GAP:
                first, second = ((a_album, ya), (b_album, yb)) if ya <= yb \
                    else ((b_album, yb), (a_album, ya))
                out.append(_bridge(
                    RELEASE_YEAR_GAP, "both", [ra, rb],
                    text=(f"{first[0]} came out in {first[1]} and "
                          f"{second[0]} in {second[1]}, "
                          f"{number_words(gap)} years apart."),
                    subjects=[subj_a, subj_b],
                    values={"years_apart": gap,
                            "years": [str(first[1]), str(second[1])],
                            "earlier": first[0], "later": second[0]},
                    objects=span_words(gap),
                    grounds=[str(ya), str(yb), f"{gap} years",
                             f"{number_words(gap)} years", "apart",
                             "came out", "released", "years apart",
                             "decades"]))
            elif ya // 10 == yb // 10:
                decade = f"{ya // 10 * 10}s"
                out.append(_bridge(
                    SAME_DECADE, "both", [ra, rb],
                    text=(f"{a_album} came out in {ya} and {b_album} in "
                          f"{yb}."),
                    subjects=[subj_a, subj_b],
                    values={"decade": decade, "years": [str(ya), str(yb)]},
                    objects=[decade], grounds=[str(ya), str(yb)],
                    eligible=False, reason=SAME_DECADE_NOT_A_POINT))
            else:
                out.append(_bridge(
                    RELEASE_YEAR_GAP, "both", [ra, rb],
                    text=(f"{a_album} came out in {ya} and {b_album} in "
                          f"{yb}."),
                    subjects=[subj_a, subj_b],
                    values={"years_apart": gap, "years": [str(ya), str(yb)]},
                    objects=[str(gap)], grounds=[str(ya), str(yb)],
                    eligible=False, reason=GAP_TOO_SMALL))
    if same_act:
        # the same act on both sides: the run is the story, and everything
        # below would compare an artist with itself
        return out
    # producer ----------------------------------------------------------------
    for pa in _all(a, "album_producer"):
        for pb in _all(b, "album_producer"):
            if pa.get("fact_id") == pb.get("fact_id"):
                continue
            if _same(pa.get("value"), pb.get("value")) and \
                    not _same(a_album, b_album):
                producer = str(pa.get("value") or "")
                self_made = (_contains(a_artist, producer)
                             or _contains(b_artist, producer))
                out.append(_bridge(
                    SAME_PRODUCER, "both", [pa, pb],
                    text=(f"{a_album} and {b_album} were both produced by "
                          f"{producer}."),
                    subjects=[subj_a, subj_b],
                    values={"producer": producer},
                    objects=_tokens(producer),
                    grounds=[producer, "produced by", "both produced"],
                    eligible=not self_made,
                    reason=SELF_PRODUCED if self_made else None))
                break
    # shared member -----------------------------------------------------------
    members_b = {}
    for fb in _all(b, "band_member"):
        mb = str(_meta(fb).get("member_mbid") or "")
        if mb:
            members_b.setdefault(mb, fb)
    for fa in _all(a, "band_member"):
        ma = str(_meta(fa).get("member_mbid") or "")
        fb = members_b.get(ma) if ma else None
        if fb is None:
            continue
        member = str(_meta(fa).get("member_name") or fa.get("value") or "")
        if not member:
            continue
        out.append(_bridge(
            SHARED_MEMBER, "both", [fa, fb],
            text=f"{member} played in both {a_artist} and {b_artist}.",
            subjects=[art_a, art_b],
            values={"member": member},
            objects=_tokens(member),
            grounds=[member, "played in both", "played in"]))
        break                                # one name is a line; five is a roster
    # origin --------------------------------------------------------------------
    oa = next((f for f in a if f.get("predicate") in _ORIGIN), None)
    ob = next((f for f in b if f.get("predicate") in _ORIGIN), None)
    if oa and ob and oa.get("fact_id") != ob.get("fact_id"):
        place_a, place_b = _place(oa), _place(ob)
        if _same(place_a, place_b):
            generic = _country_level(oa) or _country_level(ob)
            out.append(_bridge(
                SAME_ORIGIN, "both", [oa, ob],
                text=f"{a_artist} and {b_artist} both came out of {place_a}.",
                subjects=[art_a, art_b],
                values={"place": place_a},
                objects=_tokens(place_a),
                grounds=[place_a, "both came out of", "came out of"],
                eligible=not generic,
                reason=GENERIC_COUNTRY if generic else None))
    # formed the same year --------------------------------------------------------
    fa_, fb_ = _one(a, "group_formed_year"), _one(b, "group_formed_year")
    if fa_ and fb_ and fa_.get("fact_id") != fb_.get("fact_id"):
        ya, yb = _year(fa_.get("value")), _year(fb_.get("value"))
        if ya and yb and ya == yb:
            out.append(_bridge(
                SAME_FORMED_YEAR, "both", [fa_, fb_],
                text=f"{a_artist} and {b_artist} both formed in {ya}.",
                subjects=[art_a, art_b],
                values={"year": str(ya)}, objects=[str(ya)],
                grounds=[str(ya), "both formed", "formed in"]))
    # label: derived for the record, never offered ----------------------------------
    la, lb = _one(a, "album_record_label"), _one(b, "album_record_label")
    if la and lb and la.get("fact_id") != lb.get("fact_id") and \
            _same(la.get("value"), lb.get("value")) and \
            not _same(a_album, b_album):
        label = str(la.get("value") or "")
        out.append(_bridge(
            SAME_LABEL, "both", [la, lb],
            text=f"{a_album} and {b_album} both came out on {label}.",
            subjects=[subj_a, subj_b], values={"label": label},
            objects=_tokens(label), grounds=[label],
            eligible=False, reason=LABEL_BRIDGES_DISABLED))
    # recording place, where a pack supplies it (today's packs do not) --------------
    for kind, preds in _RECORDING_PLACE.items():
        pa = next((f for f in a if f.get("predicate") in preds), None)
        pb = next((f for f in b if f.get("predicate") in preds), None)
        if pa and pb and pa.get("fact_id") != pb.get("fact_id") and \
                _same(pa.get("value"), pb.get("value")) and \
                not _same(a_album, b_album):
            place = str(pa.get("value") or "")
            where = ("at" if kind == SAME_RECORDING_STUDIO else "in")
            out.append(_bridge(
                kind, "both", [pa, pb],
                text=(f"{a_album} and {b_album} were both recorded {where} "
                      f"{place}."),
                subjects=[subj_a, subj_b], values={"place": place},
                objects=_tokens(place),
                grounds=[place, "recorded", "both recorded"]))
    return out


def _side_bridges(pool: list[dict[str, Any]], side: dict[str, Any],
                  about: str, today: date | None) -> list[dict[str, Any]]:
    """Relationships INSIDE one act's own story, and against the calendar."""
    out: list[dict[str, Any]] = []
    artist, album = _side_names(side)
    subj = _subject_alts(album, artist)
    art = _subject_alts(artist)
    # a self-titled record is said as such, never "Cold Orchard came out"
    album_said = (f"{artist}'{'' if artist.endswith('s') else 's'} "
                  "self-titled album" if _same(album, artist) else album)
    rel = _release(pool)
    formed = _one(pool, "group_formed_year")
    # career span: the band formed, and this record came out much later
    if rel and formed and artist and album:
        fy, ry = _year(formed.get("value")), _year(rel.get("value"))
        if fy and ry and ry > fy:
            span = ry - fy
            if span >= MIN_CAREER_SPAN:
                out.append(_bridge(
                    CAREER_SPAN, about, [formed, rel],
                    text=(f"{artist} formed in {fy}, and {album_said} came "
                          f"out {number_words(span)} years later, in {ry}."),
                    subjects=[art],
                    values={"formed": str(fy), "released": str(ry),
                            "years": span},
                    objects=span_words(span),
                    grounds=[str(fy), str(ry), f"{span} years",
                             f"{number_words(span)} years", "formed",
                             "came out", "released", "later",
                             "years later", "decades"]))
            else:
                out.append(_bridge(
                    CAREER_SPAN, about, [formed, rel],
                    text=(f"{artist} formed in {fy}; {album} came out in "
                          f"{ry}."),
                    subjects=[art],
                    values={"formed": str(fy), "released": str(ry),
                            "years": span},
                    objects=[str(span)], grounds=[str(fy), str(ry)],
                    eligible=False, reason=SPAN_TOO_SHORT))
    if today is None:
        return out
    # release anniversary / recent release
    if rel and album:
        full = _full_date(rel.get("value"))
        ry = _year(rel.get("value"))
        if full and (full.month, full.day) == (today.month, today.day) \
                and today.year > full.year:
            n = today.year - full.year
            out.append(_bridge(
                RELEASE_ANNIVERSARY, about, [rel],
                text=(f"{album_said} came out {number_words(n)} years ago "
                      f"today, on {full.strftime('%B')} {full.day}, "
                      f"{full.year}."),
                subjects=[subj],
                values={"years_ago": n, "released": full.isoformat()},
                objects=[str(n), number_words(n), str(full.year)],
                grounds=[str(full.year), f"{n} years", f"{number_words(n)} "
                         "years", "ago today", "years ago", "came out",
                         "released", "anniversary"]))
        elif ry and 0 <= today.year - ry <= RECENT_RELEASE_YEARS:
            when = "this year" if ry == today.year else "last year"
            out.append(_bridge(
                RECENT_RELEASE, about, [rel],
                text=f"{album_said} came out {when}, in {ry}.",
                subjects=[subj],
                values={"released": str(ry), "when": when},
                objects=[str(ry), when],
                grounds=[str(ry), when, "came out", "released", "new",
                         "latest"]))
    # birthday (a person, and not a memorial)
    born = _one(pool, "date_of_birth")
    if born and artist and not _one(pool, "date_of_death"):
        bd = _full_date(born.get("value"))
        if bd and (bd.month, bd.day) == (today.month, today.day) \
                and today.year > bd.year:
            age = today.year - bd.year
            out.append(_bridge(
                BIRTHDAY, about, [born],
                text=f"{artist} turns {age} today.",
                subjects=[art], values={"age": age, "born": bd.isoformat()},
                # "it's her birthday today" is the whole fact; the age is
                # a detail the host may add
                objects=[str(age), number_words(age), "birthday"],
                grounds=[str(age), number_words(age), "today", "birthday",
                         "born", "years old", "year old"]))
    return out


def derive_bridges(ending_pool: list[dict[str, Any]] | None,
                   next_pool: list[dict[str, Any]] | None,
                   ending: dict[str, Any] | None,
                   nxt: dict[str, Any] | None,
                   today: date | datetime | None = None
                   ) -> list[dict[str, Any]]:
    """Every relationship the RAW pools support, eligible or not, ranked.

    `ending_pool` / `next_pool` are the scope-safe facts about each side
    (the retriever's pre-relevance pool: raw knowledge stays available
    for reasoning). `ending` / `nxt` carry the canonical artist and album
    the match resolved. `today` is the station's local date, for
    anniversaries; None disables the calendar bridges."""
    if isinstance(today, datetime):
        today = today.date()
    a = [f for f in (ending_pool or []) if isinstance(f, dict)]
    b = [f for f in (next_pool or []) if isinstance(f, dict)]
    sa, sb = dict(ending or {}), dict(nxt or {})
    out: list[dict[str, Any]] = []
    same_act = bool(sa.get("artist")) and _same(sa.get("artist"),
                                                sb.get("artist"))
    if a and sa.get("artist"):
        out.extend(_side_bridges(a, sa, "ending", today))
    if b and sb.get("artist"):
        out.extend(_side_bridges(b, sb, "next", today))
    if a and b and sa.get("artist") and sb.get("artist"):
        out.extend(_pair_bridges(a, b, sa, sb, same_act))
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for br in sorted(out, key=lambda x: (-x["score"], x["id"])):
        if br["id"] in seen:
            continue
        seen.add(br["id"])
        uniq.append(br)
    return uniq


# --- the offer --------------------------------------------------------------------

def compose(eligible: list[dict[str, Any]] | None,
            bridges: list[dict[str, Any]] | None, *,
            member_names: set[str] | None = None) -> dict[str, Any]:
    """The writer's optional enrichment: {'facts': [...], 'diagnostics'}.

    `eligible` are the relevance-approved facts (fact_relevance.select
    already ran; each carries `relation`) in retrieval rank order.
    `bridges` are derive_bridges' output. The offer is bridges first
    (at most MAX_BRIDGES, one per type), then dj_ready facts (at most
    MAX_DJ_READY, one per predicate, the next track contributing at most
    one), never more than MAX_OFFER in all. Everything held back is
    reported with its class and reason. Empty is a valid, good answer.
    """
    facts = [f for f in (eligible or []) if isinstance(f, dict)]
    brs = [b for b in (bridges or []) if isinstance(b, dict)]
    offered: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    types_seen: set[str] = set()
    consumed: set[str] = set()
    for br in brs:
        if not br.get("eligible"):
            continue
        if br["bridge_type"] in types_seen or len(offered) >= MAX_BRIDGES:
            continue
        if consumed & set(br.get("evidence") or []):
            # two relationships built on the same row are one point made
            # twice; the stronger one has already been taken
            continue
        types_seen.add(br["bridge_type"])
        offered.append({k: v for k, v in br.items()
                        if k not in ("eligible", "reason")})
        consumed.update(br.get("evidence") or [])
    classes: dict[tuple[str, str, str, str], int] = {}

    def hold(f: dict[str, Any], cls: str, why: str | None) -> None:
        key = (str(f.get("predicate")), str(f.get("about")), cls,
               str(why or ""))
        classes[key] = classes.get(key, 0) + 1
        held.append({"id": f.get("id"), "predicate": f.get("predicate"),
                     "about": f.get("about"), "class": cls,
                     "suppression_reason": why})

    ready = 0
    preds: set[str] = set()
    next_used = False
    # editorial order among dj_ready facts, the ending side leading and
    # retrieval rank breaking ties; classification is decided per fact
    ranked = sorted(
        enumerate(facts),
        key=lambda it: (0 if it[1].get("about") != "next" else 1,
                        -_DJ_READY_RANK.get(str(it[1].get("predicate")), 0.0),
                        it[0]))
    for _, f in ranked:
        cls, why = classify(f, member_names)
        if cls != DJ_READY:
            hold(f, cls, why)
            continue
        if f.get("fact_id") in consumed:
            hold(f, cls, FOLDED_INTO_BRIDGE)
            continue
        if (ready >= MAX_DJ_READY or len(offered) >= MAX_OFFER
                or f.get("predicate") in preds
                or (f.get("about") == "next" and next_used)):
            hold(f, cls, OVER_BUDGET)
            continue
        preds.add(str(f.get("predicate")))
        if f.get("about") == "next":
            next_used = True
        ready += 1
        offered.append({**f, "class": DJ_READY})
    diag = {
        "raw_bridges": len(brs),
        # the ledger names evidence by its short packet id; the OFFERED
        # rows below, and the forensic decision record, carry the full
        # pack fact_ids
        "bridges": [{"type": b["bridge_type"], "about": b.get("about"),
                     "evidence": [_short(e) for e in (b.get("evidence")
                                                     or [])],
                     "values": dict(b.get("values") or {}),
                     "eligible": bool(b.get("eligible")),
                     "reason": b.get("reason"),
                     "offered": any(o["id"] == b["id"] for o in offered)}
                    for b in brs[:6]],
        "classes": sorted(
            ({"predicate": p, "about": a, "class": c,
              "suppression_reason": r or None, "count": n}
             for (p, a, c, r), n in classes.items()),
            key=lambda d: (-d["count"], d["predicate"]))[:8],
        "offered": [{"id": o["id"],
                     "kind": "bridge" if o.get("kind") == "bridge" else "fact",
                     "predicate": o.get("predicate"),
                     "about": o.get("about"),
                     "evidence": list(o.get("evidence") or [])}
                    for o in offered],
    }
    # REFERENCE material for the validator alone: the release years and
    # formed years held back as bridge material. Never shown to the
    # writer -- but a writer that reaches into its own memory for "Brass
    # Weather, from 2002" and gets it right has said something the pack can
    # vouch for, and the claim scanner needs the row to vouch with.
    reference: list[dict[str, Any]] = []
    seen_ref: set[tuple[str, str]] = set()
    for f in facts:
        if classify(f, member_names)[0] != BRIDGE_CANDIDATE:
            continue
        key = (str(f.get("predicate")), str(f.get("about")))
        if key in seen_ref:
            continue
        # a row that is also evidence of an offered bridge stays here too:
        # "Northern Static dropped back in 2009" beside a career-span
        # bridge is a correct mention, and the hidden row is the only
        # thing that can vouch for it when the bridge itself was not made
        seen_ref.add(key)
        reference.append(f)
        if len(reference) >= MAX_REFERENCE:
            break
    return {"facts": offered[:MAX_OFFER], "suppressed": held,
            "reference": reference, "diagnostics": diag}
