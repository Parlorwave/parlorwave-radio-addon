"""Why a true fact belongs in THIS transition -- and when it does not.

A fact being TRUE is not sufficient reason to say it. A station can have
grounding under control and still sound like a database reading itself
out: an obscure session bassist because the pack holds personnel rows, a
record label because a label row exists, a band member from an era nobody
mentioned. Every one of those can be true, scope-safe and correctly
attributed. None of them answers the only question a break
actually asks:

    why does this belong between THESE two songs?

So this module sits between retrieval and the writer and answers that
question deterministically, BEFORE the model is asked to compose
anything. Parlorwave owns the relevance decision for the same reason it
owns the match: handing a small writer model a grab bag of unrelated true things
and hoping it picks well is how "information barf" happens, and a
negative prompt rule cannot undo an offered opportunity.

TWO GATES, and they are different questions.

  RELATION    what this fact has to do with this boundary at all, from
              authoritative structure only -- the roles the scheduler
              resolved, the album/artist run it detected, and the fact's
              own subject and spoken object. Never latent model
              knowledge, never "it was in the packet", and never mere
              recency: an artist that played four songs ago has no
              relation to the transition happening now.

  STANDING    whether that relation is ENOUGH for this KIND of fact. A
              release date for the album that just played is about the
              music in the room. A name from the band's roster is not --
              the pack carries many `band_member` rows with a constant
              interestingness of 0.75 and no era, no instrument and no
              notability, so there is no structural way to tell a
              founding member from a session player who passed through in
              1994. A fact class with no way to earn its place gets one:
              a documented link to the other side of the transition, a
              title reference, a run, or a listener request.

WHAT THIS MODULE NEVER DOES. It never invents a relationship, never
consults a model, never reads a database, and never promotes a fact it
would otherwise drop in order to keep the packet full. When nothing has
a defensible relation the right answer is fewer facts -- or none, which
is a perfectly good break.

Single-anchor doctrine is preserved: a fact about the ENDING
artist is eligible without mentioning what is next, a fact about the NEXT
artist is eligible without mentioning what just played, and no fact at
all is a valid enrichment set.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# --- relations --------------------------------------------------------------------
#: The documented reasons a fact may be in this break, strongest first.
#: The name travels into forensics, so a log reader can ask "which
#: reason put this sentence on air" and get an answer.
CURRENT_NEXT_RELATIONSHIP = "current_next_relationship"
TITLE_REFERENCE = "title_reference"
SAME_ARTIST_RUN = "same_artist_run"
SAME_ALBUM_RUN = "same_album_run"
LISTENER_REQUEST = "listener_request"
CURRENT_TRACK = "current_track"
CURRENT_ARTIST = "current_artist"
NEXT_TRACK = "next_track"
NEXT_ARTIST = "next_artist"

RELATIONS = (CURRENT_NEXT_RELATIONSHIP, TITLE_REFERENCE, SAME_ARTIST_RUN,
             SAME_ALBUM_RUN, LISTENER_REQUEST, CURRENT_TRACK, CURRENT_ARTIST,
             NEXT_TRACK, NEXT_ARTIST)

#: A relation that is an EDITORIAL EVENT in its own right -- the station
#: did something, or the two sides are documented to be connected. These
#: carry a fact class that plain aboutness cannot.
STRONG = frozenset({CURRENT_NEXT_RELATIONSHIP, TITLE_REFERENCE,
                    SAME_ARTIST_RUN, SAME_ALBUM_RUN, LISTENER_REQUEST})

# --- suppression ------------------------------------------------------------------
#: Nothing ties this fact to either side of the transition.
NO_RELATION = "no_relation"
#: It is about the right music, but its KIND needs more than that.
NEEDS_EDITORIAL_REASON = "needs_editorial_reason"
#: Eligible, but the break's fact budget was already spent on better.
OVER_BUDGET = "over_budget"

#: Fact classes that plain aboutness does not justify.
#:
#: personnel   the pack's rosters are all-time and undated: nothing in
#:             the data distinguishes a founder from a footnote, so
#:             "<name> was in the band" is offered only when that person
#:             or group is the LINK between the two songs playing.
#: reissue     provenance about a pressing, not about the music.
#: birth date  a date with no occasion attached to it.
PERSONNEL = frozenset({"band_member", "member_of_group",
                       "cross_band_membership"})
NEEDS_STRONG = PERSONNEL | frozenset({"reissue_release_date",
                                      "date_of_birth"})

#: The shortest name this module will match on. Below it a name cannot be
#: told apart from ordinary words, and a false relationship is worse than
#: a missing one. (Same bar as smart_dj.ANCHOR_NAME_MIN, kept local so
#: this module imports nothing.)
NAME_MIN = 3


def _norm(text: Any) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _same(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na) and na == nb


def _mentions(haystack: str, name: str) -> bool:
    """Does `haystack` name `name`, word-bounded? Used only to find a
    DOCUMENTED link, so it is deliberately strict: a two-letter band name
    would match half the English language."""
    n = _norm(name)
    if len(n) < NAME_MIN:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])",
                          _norm(haystack)))


def boundary(context: dict[str, Any] | None) -> dict[str, Any]:
    """The transition, as the facts about it -- built from the render
    context the scheduler already resolved. Pure: no database, no clock,
    no settings. `artist_run` is derived here because 'the same act on
    both sides' is arithmetic over two names the scheduler supplied, not
    a judgement."""
    ctx = context or {}
    pre = str(ctx.get("pre_name") or "")
    post = str(ctx.get("post_name") or "")
    ea, _, et = pre.partition(" - ") if " - " in pre else ("", "", pre)
    na, _, nt = post.partition(" - ") if " - " in post else ("", "", post)
    b = {
        "ending_artist": ea.strip(),
        "ending_title": et.strip(),
        "ending_album": str(ctx.get("ending_track_album") or "").strip(),
        "next_artist": na.strip(),
        "next_title": nt.strip(),
        "next_album": str(ctx.get("next_track_album") or "").strip(),
        "album_run": bool(ctx.get("album_run")),
        "requested_artist": str(ctx.get("requested_artist") or "").strip(),
    }
    b["artist_run"] = _same(b["ending_artist"], b["next_artist"])
    return b


def for_tracks(ending: tuple[str, str, str | None] | None,
               nxt: tuple[str, str, str | None] | None = None,
               **extra: Any) -> dict[str, Any]:
    """The same boundary from the (artist, title, album) tuples the
    knowledge retriever already takes, so a caller that has no render
    context still gets a real relevance decision rather than an
    everything-passes default."""
    e = ending or ("", "", None)
    n = nxt or ("", "", None)
    b = {
        "ending_artist": str(e[0] or "").strip(),
        "ending_title": str(e[1] or "").strip(),
        "ending_album": str(e[2] or "").strip(),
        "next_artist": str(n[0] or "").strip(),
        "next_title": str(n[1] or "").strip(),
        "next_album": str(n[2] or "").strip(),
        "album_run": bool(extra.get("album_run")),
        "requested_artist": str(extra.get("requested_artist") or "").strip(),
    }
    b["artist_run"] = _same(b["ending_artist"], b["next_artist"])
    return b


def _sides(fact: dict[str, Any], b: dict[str, Any]
           ) -> tuple[str, str, str, str]:
    """(own artist, own title, other artist, other title) for the side of
    the transition this fact is about."""
    if (fact.get("about") or "ending") == "next":
        return (b.get("next_artist") or "", b.get("next_title") or "",
                b.get("ending_artist") or "", b.get("ending_title") or "")
    return (b.get("ending_artist") or "", b.get("ending_title") or "",
            b.get("next_artist") or "", b.get("next_title") or "")


def _spoken_object(fact: dict[str, Any]) -> str:
    """What this fact's sentence actually NAMES, as one searchable string:
    the realization policy's object tokens plus the raw value. This is
    what can link a fact to the other side of the transition -- a member
    who is also the next act, a side project that is the next band."""
    realize = fact.get("realize") or {}
    parts = [str(t) for t in (realize.get("object") or [])]
    if fact.get("value"):
        parts.append(str(fact["value"]))
    return " ".join(parts)


def _album_of(fact: dict[str, Any], b: dict[str, Any]) -> str:
    """The album the boundary knows for this fact's side, or ''."""
    return (b.get("next_album") or "") if fact.get("about") == "next" \
        else (b.get("ending_album") or "")


def _linked(a: str, b: str) -> bool:
    """Do these two names refer to the same thing, generously? The pack's
    canonical spelling and the player's are rarely byte-identical ('Wren
    Hollis' / 'The Wren Hollis Experience'), so containment either way
    counts -- this test only ever CONFIRMS a relation, so a generous
    reading costs nothing a strict one would have caught."""
    return _mentions(a, b) or _mentions(b, a)


def _about_right_music(fact: dict[str, Any], b: dict[str, Any],
                       own_artist: str, own_title: str) -> bool:
    """Is this fact actually about the music on this side of the break?

    CHECKED ONLY WHERE THERE IS SOMETHING AUTHORITATIVE TO CHECK AGAINST.
    An artist-scope fact's anchor is the canonical ARTIST by construction,
    so the artist name settles it. An album- or recording-scope fact's
    anchor is the canonical ALBUM or TITLE, which the boundary knows only
    when the catalog supplied one -- so when it did not, the retriever's
    own `about` role stands. A gate that guessed here would drop true,
    relevant album facts for tracks whose catalog album is blank, which is
    a worse failure than the one it would prevent."""
    anchor = str(fact.get("anchor") or "")
    if not anchor:
        return True
    if str(fact.get("scope") or "") == "artist":
        return not own_artist or _linked(anchor, own_artist)
    album = _album_of(fact, b)
    if not album:
        return True
    return _linked(anchor, album) or bool(own_title and _linked(anchor,
                                                                own_title))


def classify(fact: dict[str, Any], b: dict[str, Any]) -> str | None:
    """The documented reason this fact belongs in this break, or None.

    Checked strongest first, so the recorded reason is the best one that
    is true rather than the first one that happens to be tested. Nothing
    here consults a model, a database or the session's history: RECENCY IS
    NOT A RELATION, and an artist that played four songs ago has no claim
    on the transition happening now."""
    if not isinstance(fact, dict) or not isinstance(b, dict):
        return None
    role = "next" if fact.get("about") == "next" else "ending"
    if (fact.get("about") or "ending") not in ("ending", "next"):
        return None            # a fact about neither side of this break
    own_artist, own_title, other_artist, other_title = _sides(fact, b)
    if not (own_artist or own_title):
        return None            # this fact is about a side that is not here
    if not _about_right_music(fact, b, own_artist, own_title):
        return None
    obj = _spoken_object(fact)
    scope = str(fact.get("scope") or "")
    if other_artist and not _same(own_artist, other_artist) \
            and _mentions(obj, other_artist):
        return CURRENT_NEXT_RELATIONSHIP
    if other_title and _mentions(obj, other_title):
        return TITLE_REFERENCE
    if b.get("artist_run"):
        return SAME_ARTIST_RUN
    if b.get("album_run") and scope != "artist":
        return SAME_ALBUM_RUN
    if b.get("requested_artist") and _same(own_artist, b["requested_artist"]):
        return LISTENER_REQUEST
    if scope == "artist":
        return NEXT_ARTIST if role == "next" else CURRENT_ARTIST
    return NEXT_TRACK if role == "next" else CURRENT_TRACK


def on_topic(fact: dict[str, Any], b: dict[str, Any]) -> bool:
    """Is this fact ABOUT the music on its side of the break at all --
    RELATION alone, with no STANDING judgement? What the editorial layer
    derives relationships from: a roster row has no standing on its own,
    but it is still about the band that is playing, whereas an album
    fact whose album is not the catalog's album is about something else
    entirely and may not seed a bridge either."""
    try:
        return classify(fact, b) is not None
    except Exception:  # noqa: BLE001
        return False


def needs_strong_reason(fact: dict[str, Any]) -> bool:
    """Is this a fact class that plain aboutness does not justify?"""
    pred = str(fact.get("predicate") or "")
    if pred in NEEDS_STRONG:
        return True
    # An ORDINARY running time is true of most songs ever recorded; only a
    # deterministically unusual one is worth a remark, and the retriever
    # is what decides which this is (`notable`).
    return pred == "recording_length" and not fact.get("notable")


def decide(fact: dict[str, Any], b: dict[str, Any]
           ) -> tuple[str | None, str | None]:
    """(relation, suppression_reason). Exactly one of the two is set."""
    relation = classify(fact, b)
    if relation is None:
        return None, NO_RELATION
    if needs_strong_reason(fact) and relation not in STRONG:
        return None, NEEDS_EDITORIAL_REASON
    return relation, None


def select(facts: list[dict[str, Any]] | None, b: dict[str, Any] | None,
           limit: int | None = None) -> dict[str, Any]:
    """{'facts': [...], 'suppressed': [...]}.

    Eligible facts keep their input order (retrieval already ranked them)
    and gain a `relation` key. Suppressed facts are reported -- with the
    reason -- because "the station chose not to offer this" is exactly
    the thing a log review cannot reconstruct afterwards.

    Never raises: a relevance fault must not cost the break its
    enrichment, so on any internal failure the facts pass through
    unchanged and the suppression list is empty.
    """
    out: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    rows = [f for f in (facts or []) if isinstance(f, dict)]
    if b is None:
        return {"facts": rows, "suppressed": []}
    try:
        for f in rows:
            relation, why = decide(f, b)
            if relation is None:
                suppressed.append({"id": f.get("id"),
                                   "predicate": f.get("predicate"),
                                   "about": f.get("about"),
                                   "suppression_reason": why})
                continue
            if limit is not None and len(out) >= max(0, int(limit)):
                suppressed.append({"id": f.get("id"),
                                   "predicate": f.get("predicate"),
                                   "about": f.get("about"),
                                   "relation": relation,
                                   "suppression_reason": OVER_BUDGET})
                continue
            out.append({**f, "relation": relation})
    except Exception:  # noqa: BLE001 - enrichment is never worth a break
        return {"facts": rows, "suppressed": []}
    return {"facts": out, "suppressed": suppressed}
