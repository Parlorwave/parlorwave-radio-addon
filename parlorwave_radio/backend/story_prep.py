"""Story preparation: turn Music Assistant's artist biographies into stories a
host can TELL -- prepared and checked before air, never improvised live.

Music Assistant knows the music; Parlorwave knows how to talk about it. The
listener's own MA usually holds a biography for almost every artist in
the library; this module reads it, has the local writer
model propose stories from it, and keeps only the ones that pass a
LIGHTWEIGHT source-consistency check. Nothing here is bundled: the material
is read from the listener's MA at preparation time and cached locally
(dj_memory).

THE SHAPE OF A STORY. A story is three parts, kept apart all the way to air:

    setup    optional framing   ("Here's how one band got its name.")
    telling  the factual core   (the anecdote, retold in plain words)
    payoff   optional framing   ("Worked out fine.")

The TELLING carries every fact and is checked against the source sentences
it cites. The setup and payoff are the host's own words -- personality --
and may add no fact at all: no name, number, place, date or event that the
telling does not already carry, no claim about anyone's feelings or
intentions, no pseudo-profundity. A story whose framing fails is kept as its
telling alone.

THE CHECKS ARE LIGHT ON PURPOSE (a design decision). A biography
anecdote may be used without independent corroboration. What is refused:

  * a name or number in the telling that its source sentences do not carry
    (invented people, places, dates);
  * a telling that drifts from its source (too few of its words come from
    the cited sentences) or drops a negation;
  * an elapsed span ("thirty years later") the source does not state -- the
    elapsed-span class: no date arithmetic;
  * stale tense ("will release her debut album next spring");
  * a contradiction of the knowledge pack where the pack is solid (formed
    year, studio-album year);
  * long verbatim runs of the biography: retell, don't copy;
  * framing that adds facts, guesses at feelings or philosophises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

#: The only Music Assistant commands this module sends. `music/artists/get`
#: schedules a metadata refresh unless told not to, so the flag is forced.
READ_COMMANDS = frozenset({"music/artists/library_items", "music/artists/get"})
NO_REFRESH = {"allow_update_metadata": False}

SOURCE_PROVIDER = "music_assistant"
SOURCE_FIELD = "artist.metadata.description"

KIND_STORY = "story"        # setup + telling + payoff, 35-75 spoken words
KIND_NUGGET = "nugget"      # one short telling for an ordinary break

STORY_WORDS = (20, 75)      # telling + framing, story kind
NUGGET_WORDS = (6, 30)
FRAMING_MAX_WORDS = 18
VERBATIM_MAX_RUN = 12       # longest run of copied prose words (names
                            # do not count: a list of acts is not prose)
CHUNK_SENTENCES = 12        # sentences shown to the model per request
MIN_SOURCE_COVERAGE = 0.35  # share of the telling's content words in source

# --- word lists -------------------------------------------------------------------
STOP = set("""a an the of in on at to for and or but with by from as is was were
are be been being has had have their his her its they he she it this that these
those who which while after before when then also into over up out off later
first one there here so just very really even still only all both any some
about through during under between against around again than too more most
such own same other pretty quite really just big lot lots ton tons
kind sort thing things stuff way back""".split())
NEG = re.compile(r"\b(not|never|no|without|nor|neither|unable|failed|"
                 r"didn'?t|wasn'?t|weren'?t|couldn'?t|wouldn'?t|refused)\b",
                 re.I)
STALE = re.compile(r"\b(will|currently|recently|upcoming|new album|is working|"
                   r"are working|plans? to|set to|later this year|next year|"
                   r"this year|is expected|are expected)\b", re.I)
SPAN = re.compile(r"\b(?:\d{1,3}|(?:a|one|two|three|four|five|six|seven|eight|"
                  r"nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|"
                  r"sixty|seventy|eighty|ninety|a couple of|a few)(?:[- ]\w+)?)\s+"
                  r"(?:years?|decades?)\s+(?:later|after|ago|before|on|apart|of)"
                  r"\b", re.I)
#: Sombre material: told respectfully if at all, never with a joke, and
#: kept OUT of everyday rotation.
SOMBRE = re.compile(r"\b(passed away|died|death|dead|suicide|overdos\w*|"
                    r"addict\w*|heroin|cancer|illness|murder\w*|killed|"
                    r"abuse\w*|rape|funeral|depress\w*|shot|stabbed|"
                    r"assault\w*|violence|violent|crash\w*|mental health)\b",
                    re.I)
SENSITIVE = re.compile(r"\b(passed away|passing|died|death|dead|suicide|overdos\w*|addict\w*|"
                       r"heroin|cancer|illness|murder\w*|killed|abuse\w*|"
                       r"rape|funeral|depress\w*)\b", re.I)
#: Framing may not do these: philosophise, guess at feelings or intent,
#: or call the music great.
FILLER = re.compile(r"|".join([
    r"something to think about", r"isn'?t it\b", r"didn'?t it\b",
    r"there'?s (a certain|something (about|to))", r"\bin a way\b",
    r"remind(s|er)? (us|me|you|that|of)", r"\bjourney\b", r"\bvibes?\b",
    r"\benergy\b", r"\btimeless\b", r"it'?s all about",
    r"makes? (you|one|me) (think|wonder|feel)", r"the beauty of",
    r"\bresonat", r"\bspeaks to\b", r"\bthe (human|soul|heart) (of|condition)",
    r"\bcontemplat", r"\bponder", r"\breflect", r"\bintrospect",
    r"\btapestry\b", r"\blandscape\b", r"\bwe all\b", r"\blife'?s\b",
    r"\bone (almost|wonders?|might|imagines?|suspects?)\b", r"\bi suppose\b",
    r"\bexistential\b", r"\bprofound\b", r"\blegacy\b", r"\bdestin(ed|y)\b",
    # stock exclamations: one is a reaction, the fifth is a tic
    r"\bpretty (cool|wild|amazing|neat|fun|incredible)\b", r"\bcool,? (huh|right)\b",
    r"^\W*wow\b", r"\bthat'?s (a lot|wild|amazing|incredible|so cool)\b",
    r"\bcan you (believe|imagine)\b", r"\bfun to (think|see|imagine)\b",
    r"\binteresting to see\b", r"\bhow things evolve\b"]),
    re.I)
#: framing that only makes sense after something else was said
NEEDS_CONTEXT = re.compile(r"^\W*(and|also|plus|again|even|then|but|meanwhile|"
                           r"anyway)\b|\b(even more|back again|another one|this "
                           r"time|the story gets|next chapter|once again)\b", re.I)
#: material that is true and dull: genre lists, tallies, critics' verdicts
DULL = re.compile(r"\b(genres?|blends?|a mix of|contralto|soprano|music videos|"
                  r"singles|sold over|copies|critics?|ranked|chart(ed|s)?|"
                  r"nominat\w+|influenc\w+|inspiration)\b", re.I)
FEELINGS = re.compile(r"\b(felt|feel|feels|wanted|meant to|intended|hoped|"
                      r"loved|hated|must have|probably|surely|i bet|"
                      r"deep down|secretly|dreamed|wished)\b", re.I)
PRAISE = re.compile(r"\b(influential|iconic|legendary|greatest|genius|"
                    r"masterpiece|groundbreaking|timeless|classic|best|"
                    r"brilliant|underrated)\b", re.I)
EVENT = re.compile(r"\b(formed|founded|recorded|released|signed|toured|joined|"
                   r"left|quit|fired|married|born|died|won|wrote|produced|"
                   r"played|met|moved|sold|broke up|reunited|named|renamed)\b",
                   re.I)


# --- data -------------------------------------------------------------------------

@dataclass
class Story:
    artist: str
    kind: str
    telling: str
    setup: str = ""
    payoff: str = ""
    sentences: list[int] = field(default_factory=list)   # cited source ids
    source_text: str = ""                                # the cited sentences
    artist_mbid: str = ""
    source_ref: str = ""                                 # MA item uri / id
    hook: str = ""                                       # why it is fun
    surprise: int = 3                                    # 1-5, the model's
    proposed_setup: str = ""                             # framing as proposed,
    proposed_payoff: str = ""                            # before any check
    support_text: str = ""                               # bio sentences behind
    #                                   a fuller name ("Teodor" for "Vance")
    ok: bool = False
    refusals: list[str] = field(default_factory=list)
    framing_dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def spoken(self) -> str:
        return " ".join(p for p in (self.setup, self.telling, self.payoff)
                        if p).strip()

    @property
    def story_id(self) -> str:
        h = hashlib.sha256(f"{self.artist}|{_norm(self.telling)}".encode())
        return "st:" + h.hexdigest()[:12]

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out["story_id"] = self.story_id
        out["spoken"] = self.spoken
        out["words"] = len(self.spoken.split())
        out["passage_hash"] = hashlib.sha256(
            self.source_text.encode()).hexdigest()[:16]
        out["source_provider"] = SOURCE_PROVIDER
        out["source_field"] = SOURCE_FIELD
        # selection rank: how surprising, and whether it is a real story
        out["sombre"] = bool(SOMBRE.search(self.spoken))
        out["list_shaped"] = list_shaped(self.telling)
        out["quality"] = round(0.2 + 0.12 * self.surprise
                               + (0.1 if self.kind == KIND_STORY else 0.0)
                               # a dull fact (genre blends, chart and sales
                               # counts, "influences") is worse than no story:
                               # always below dj_memory.MIN_QUALITY
                               - (0.55 if DULL.search(self.telling) else 0.0)
                               - (0.2 if out["list_shaped"] else 0.0), 3)
        return out


# --- reading Music Assistant ------------------------------------------------------

def ma_read(api_fn: Callable[[str, dict[str, Any]], Any], command: str,
            args: dict[str, Any]) -> Any:
    """One allowlisted, refresh-free Music Assistant read."""
    if command not in READ_COMMANDS:
        raise ValueError(f"refusing Music Assistant command {command}")
    args = dict(args)
    if command == "music/artists/get":
        args.update(NO_REFRESH)
    return api_fn(command, args)


def library_artists(api_fn, page: int = 500, bound: int = 5000
                    ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while len(out) < bound:
        got = ma_read(api_fn, "music/artists/library_items",
                      {"limit": page, "offset": len(out)})
        items = got.get("items", got) if isinstance(got, dict) else got
        if not items:
            break
        out.extend(items)
        if len(items) < page:
            break
    return out


def artist_material(api_fn, item: dict[str, Any]) -> dict[str, Any]:
    """{'artist', 'mbid', 'bio', 'genres', 'uri'} for one library artist."""
    full = ma_read(api_fn, "music/artists/get",
                   {"item_id": str(item.get("item_id")),
                    "provider_instance_id_or_domain": "library"}) or {}
    md = full.get("metadata") or {}
    mbid = next((e[1] for e in full.get("external_ids") or []
                 if e and e[0] == "musicbrainz_artistid"), "")
    return {"artist": full.get("name") or item.get("name") or "",
            "mbid": mbid, "bio": md.get("description") or "",
            "genres": list(md.get("genres") or []),
            "uri": full.get("uri") or item.get("uri") or ""}


# --- text helpers -----------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()
                  .replace("’", "'")).strip()


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"['’]s\b", "",
                                            str(text or "").lower()))


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _words(text) if w not in STOP and len(w) > 2}


def sentences(bio: str) -> list[str]:
    flat = re.sub(r"\s+", " ", str(bio or "")).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“'(])", flat)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def chunks(sents: list[str], size: int = CHUNK_SENTENCES
           ) -> list[tuple[int, list[str]]]:
    """The WHOLE biography, in consecutive windows: (first index, window)."""
    return [(i, sents[i:i + size]) for i in range(0, len(sents), size)]


#: frequent words that open spoken sentences; a capitalised one is not a name
COMMON_WORDS = set("""sadly eventually early besides meanwhile later soon finally
initially originally afterwards afterward instead still today tonight around
apparently actually basically honestly luckily unfortunately fortunately
together soon then plus also however though although because since while
during after before when where what who which why how it's that's there's
here's they they'd he'd she'd we'd i'd you'd people fans someone everyone
guitarist drummer bassist singer vocalist keyboardist producer director
critics his her their its our your""".split())
#: everyday abbreviations a retelling may use for the source's words
ALIASES = {"tv": ("television",), "us": ("united states", "u s", "america"),
           "usa": ("united states",), "uk": ("united kingdom", "britain",
                                               "england"),
           "la": ("los angeles",), "nyc": ("new york",), "dj": ("disc jockey",),
           "michigan": ("mi",), "california": ("ca",), "texas": ("tx",),
           "washington": ("wa",), "massachusetts": ("ma",),
           "grammys": ("grammy",), "grammy": ("grammys",)}
_NUM_WORD = re.compile(r"\b(?:19|20)\d\d\b|\b\d+\b")


def numbers(text: str) -> set[str]:
    return set(_NUM_WORD.findall(str(text or "").replace(",", "")))


_STOP_CAPS = {"The", "A", "An", "In", "On", "At", "After", "Before", "When",
              "While", "Their", "His", "Her", "They", "He", "She", "It", "Its",
              "This", "That", "Both", "Back", "Later", "Then", "Once", "As",
              "With", "From", "By", "For", "Of", "And", "But", "Not", "Only",
              "Even", "Just", "During", "Within", "Over", "Under", "So",
              "Here", "There", "What", "Who", "Why", "How", "Well", "Now",
              "Turns", "Get", "Picture", "Apparently", "Anyway", "True",
              "Believe", "Fun", "Story", "Worked", "Not", "No", "Yes", "Next",
              "Up", "If", "Some", "One", "Two", "Three", "First", "Second",
              # sentence openers of spoken framing (a question, an interjection)
              "Can", "Could", "Did", "Does", "Do", "Sounds", "Talk", "Imagine",
              "Guess", "Wow", "Oh", "Okay", "Yeah", "Right", "That's",
              "There's", "It's", "You", "Your", "We", "I", "I'd", "I'm",
              "Let's", "Nothing", "Never", "Always", "Maybe", "Pretty", "Quite",
              "Plus", "Also", "Still", "Yet", "Good", "Great", "Nice", "Fair",
              "Hard", "Easy", "Talking", "Speaking", "Bet", "Must", "Some",
              "Everyone", "Nobody", "Somebody", "All", "Lucky", "Funny",
              "Classic", "Gotta", "Not", "Bold", "Smart", "Strange", "Odd"}


def names(text: str, *, lenient_openers: bool = False,
          common: set[str] | None = None) -> set[str]:
    """Capitalised tokens that may be names (common sentence openers
    excepted). With lenient_openers -- for the host's own framing, which
    opens with anything ("Ever wonder...?") -- a sentence's FIRST word counts
    only when the next word is capitalised too ("Ines Marlo ...")."""
    out: set[str] = set()
    for sent in re.split(r"(?<=[.!?])\s+", str(text or "")):
        lead = re.match(r"[\s\"“'(*]*([A-Z][\w'’&.-]*)(\s+(\S+))?", sent)
        skip = None
        if lead:
            follower = (lead.group(3) or "")
            opener = _norm(re.sub(r"['’](?:s|d|ll|re|ve)$", "", lead.group(1)))
            if lenient_openers and not follower[:1].isupper():
                skip = lead.start(1)
            elif common and opener in common:
                # an ordinary word that happens to open the sentence
                # ("Sadly", "Eventually"): the biography uses it lowercase
                skip = lead.start(1)
        for m in re.finditer(r"\b[A-Z][\w'’&.-]*", sent):
            if skip is not None and m.start() == skip:
                continue
            w = re.sub(r"['’](?:s|d|ll|re|ve)$", "", m.group(0)).strip(".'’")
            if w and w not in _STOP_CAPS and len(w) > 1:
                out.add(_norm(w))
    return out


def _prose_words(text: str, tag: str) -> list[str]:
    """Lower-case words, with every capitalised token (a name) replaced by
    a marker that matches nothing, so copied NAMES never make a run."""
    out = []
    for i, tok in enumerate(re.findall(r"[A-Za-z0-9’']+", str(text or ""))):
        out.append(f"\x00{tag}{i}" if tok[:1].isupper() else tok.lower())
    return out


def longest_shared_run(text: str, source: str) -> int:
    """Longest run of consecutive prose words the text copies."""
    a, b = _prose_words(text, "a"), _prose_words(source, "b")
    if not a or not b:
        return 0
    best = 0
    index: dict[str, list[int]] = {}
    for j, w in enumerate(b):
        index.setdefault(w, []).append(j)
    for i, w in enumerate(a):
        for j in index.get(w, ()):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


# --- the checks -------------------------------------------------------------------

def check_telling(st: Story, bio: str, pack: dict[str, Any] | None = None
                  ) -> list[str]:
    """Refusal reasons for the factual core; [] when it may air."""
    why: list[str] = []
    src = st.source_text
    tell = st.telling.replace("’", "'")
    if not src:
        return ["no_source_sentence"]
    if not tell or len(tell.split()) < 4:
        why.append("empty_telling")
    if not re.search(r"[.!?][\"”’']?$", tell):
        why.append("incomplete")
    allowed_names = names(src) | names(st.artist) | {_norm(st.artist)}
    # the same PERSON by a fuller name: "Teodor" when the sentence says
    # "Vance" and the biography elsewhere says "Teodor Vance"
    for full in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z'’]+)+", bio or ""):
        toks = {_norm(t) for t in full.split()}
        if toks & allowed_names:
            allowed_names |= toks
    lowercase = {w for w in re.findall(r"\b[a-z][a-z'’]+\b", bio or "")} | {
        w.lower() for w in _STOP_CAPS} | COMMON_WORDS
    src_low = " " + _norm(src) + " "
    for n in names(tell, common=lowercase):
        if n in allowed_names or any(n in a for a in allowed_names):
            continue
        if f" {n} " in src_low:          # the source has it, lower-cased
            continue
        if any(f" {_norm(full)} " in src_low for full in ALIASES.get(n, ())):
            continue                     # "TV" for "television", "MI" ...
        why.append(f"name_not_in_source:{n}")
    extra = numbers(tell) - numbers(src)
    if extra:
        why.append("number_not_in_source:" + ",".join(sorted(extra)))
    have = content_words(src) | content_words(st.artist)
    mine = content_words(tell)
    if mine:
        covered = len(mine & have) / len(mine)
        if covered < MIN_SOURCE_COVERAGE:
            why.append(f"drifts_from_source:{covered:.2f}")
    if NEG.search(src) and not NEG.search(tell):
        # a dropped negation can invert the story; only when the telling
        # shares the negated clause's words is it a problem -- be strict
        neg_clause = next((c for c in re.split(r"[,;.]", src)
                           if NEG.search(c)), "")
        if len(content_words(neg_clause) & mine) >= 2:
            why.append("dropped_negation")
    for m in SPAN.finditer(tell):
        if _norm(m.group(0)) not in _norm(src):
            why.append("date_arithmetic:" + m.group(0).lower())
    if STALE.search(tell):
        why.append("stale_tense:" + STALE.search(tell).group(0).lower())
    if PRAISE.search(tell):
        why.append("praise:" + PRAISE.search(tell).group(0).lower())
    if longest_shared_run(tell, bio) > VERBATIM_MAX_RUN:
        why.append("verbatim_copy")
    why.extend(pack_disagreements(tell, pack or {}))
    return why


def pack_disagreements(text: str, pack: dict[str, Any]) -> list[str]:
    """Contradictions of the knowledge pack where the pack is solid:
    the formed year, and a STUDIO album's year (music_knowledge)."""
    out: list[str] = []
    years = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", text)]
    if not years:
        return out
    formed = pack.get("formed")
    artist = re.escape(str(pack.get("artist") or "").strip()) or "\b\B"
    m = re.search(rf"(?:{artist}|\bthey|\bthe band|\bthe group)\s+(?:\w+\s+)?"
                  r"(?:formed|founded|got together)\b[^.;]{0,30}?\b((?:19|20)\d\d)\b",
                  text, re.I)
    if formed and m and int(m.group(1)) != formed:
        out.append(f"formed_year_disagrees:pack={formed}")
    for title, year in (pack.get("studio_albums") or {}).items():
        key = re.escape(str(title).strip())
        if len(title.strip()) <= 3 or _norm(title) == _norm(pack.get("artist")):
            continue
        # "Northern Static (2009)", "Northern Static, released in 2009", "2009's Northern Static"
        pat = (rf"{key}[*_\"”’']*[\s,]*(?:\(|(?:which\s+)?(?:was\s+)?"
               rf"(?:released|came out|out)?\s*(?:in|from)\s+)((?:19|20)\d\d)"
               rf"|((?:19|20)\d\d)(?:['’]s)\s+(?:album\s+)?[*_\"“‘']*{key}")
        for hit in re.finditer(pat, text, re.I):
            said = int(hit.group(1) or hit.group(2))
            if said != year:
                out.append(f"album_year_disagrees:{title}={year}")
                break
    return out


def check_framing(part: str, st: Story) -> str | None:
    """None when this setup/payoff may air with the telling, else why."""
    if not part:
        return None
    t = part.replace("’", "'")
    if len(t.split()) > FRAMING_MAX_WORDS:
        return "framing_too_long"
    for rx, label in ((FILLER, "filler"), (FEELINGS, "feelings"),
                      (PRAISE, "praise")):
        m = rx.search(t)
        if m:
            return f"{label}:{m.group(0).lower()}"
    if "…" in t or "..." in t:
        return "ellipsis"
    if NEEDS_CONTEXT.search(t):
        return "needs_context"
    if numbers(t) - numbers(st.telling):
        return "framing_adds_number"
    extra = (names(t, lenient_openers=True) - names(st.telling)
             - names(st.artist) - {_norm(st.artist)})
    if extra:
        return "framing_adds_name:" + ",".join(sorted(extra))
    m = EVENT.search(t)
    if m and _stem(m.group(0).lower()) not in {_stem(w) for w in
                                               _words(st.telling)}:
        return "framing_adds_event:" + m.group(0).lower()
    if SENSITIVE.search(st.telling):
        return "no_jokes_on_a_sensitive_story"
    return None


def trim_telling(telling: str) -> tuple[str, list[str]]:
    """Drop the telling's own filler sentences -- ones that carry no name,
    no number and match the framing filler ("The band's name is pretty cool,
    right?"). The facts stay; the tic goes."""
    kept, dropped = [], []
    for sent in re.split(r"(?<=[.!?])\s+", telling.strip()):
        t = sent.replace("’", "'")
        if FILLER.search(t) and not numbers(t) \
                and not names(t, lenient_openers=True):
            dropped.append(sent)
        else:
            kept.append(sent)
    return " ".join(kept).strip(), dropped


def restrain_exclamations(st: Story) -> None:
    """No "!" anywhere in a story: on an expressive voice clone an
    exclamation makes a story break shout where real lines never do, and
    a payoff allowed one still puts a "!" in some breaks. Every part says
    it plainly."""
    for part in ("telling", "setup", "payoff"):
        setattr(st, part, re.sub(r"\s*!+", ".", getattr(st, part)))


def list_shaped(text: str) -> bool:
    """Four or more names in a row, or a long name-dense telling: harder to
    follow by ear and where speech fails (measurably more ASR misses per
    word than narrative tellings). Density is judged
    only past LIST_DENSITY_MIN_WORDS: "Cold Orchard named Gullwing after a
    ferry." is a story, not a roll call."""
    seq = re.findall(r"(?:[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)"
                     r"(?:\s*,\s*|\s+and\s+)", str(text or "") + " , ")
    words = len(str(text or "").split())
    caps = len(re.findall(r"\b[A-Z][a-z]+", str(text or "")))
    return len(seq) >= 4 or (words >= LIST_DENSITY_MIN_WORDS
                             and caps * 10 / words >= 3.5)


LIST_DENSITY_MIN_WORDS = 20


def check(st: Story, bio: str, pack: dict[str, Any] | None = None) -> Story:
    """Run every check; framing that fails is dropped, the telling decides."""
    restrain_exclamations(st)
    st.telling, trimmed = trim_telling(st.telling)
    if trimmed:
        st.notes.append("trimmed:" + " | ".join(trimmed)[:200])
    st.refusals = check_telling(st, bio, pack)
    if NEEDS_CONTEXT.search(st.telling.replace("’", "'")):
        # "Wren Hollis is back again! This time, ..." needs what came before it
        st.refusals.append("needs_context")
    st.proposed_setup = st.proposed_setup or st.setup
    st.proposed_payoff = st.proposed_payoff or st.payoff
    for attr in ("setup", "payoff"):
        why = check_framing(getattr(st, attr), st)
        if why:
            st.framing_dropped.append(f"{attr}:{why}")
            setattr(st, attr, "")
    if st.kind == KIND_NUGGET and len(st.telling.split()) > NUGGET_WORDS[1]:
        st.kind = KIND_STORY          # long enough to be told as a story
        st.notes.append("promoted_to_story")
    if st.kind == KIND_STORY and not st.refusals:
        lo, hi = STORY_WORDS
        n = len(st.spoken.split())
        if n > hi:
            st.refusals.append(f"too_long:{n}")
        elif n < lo:
            # a story whose framing fell away and is now short airs as a
            # nugget -- still true, just smaller
            st.kind = KIND_NUGGET
            st.notes.append("demoted_to_nugget")
    if st.kind == KIND_NUGGET and not st.refusals:
        if len(st.telling.split()) > NUGGET_WORDS[1]:
            st.refusals.append("nugget_too_long")
        elif SENSITIVE.search(st.telling):
            # a death notice is not a fun fact between two songs
            st.refusals.append("sensitive_fact_alone")
    st.ok = not st.refusals
    st.support_text = name_support(st, bio) if st.ok else ""
    return st


def name_support(st: Story, bio: str) -> str:
    """The biography sentences that carry a name the telling uses but its
    cited sentences do not -- the fuller-name allowance in check_telling --
    so every aired name traces to stored source text."""
    have = names(st.source_text) | names(st.artist) | {_norm(st.artist)}
    out: list[str] = []
    for n in sorted(names(st.telling) - have):
        for sent in sentences(bio):
            if re.search(rf"\b{re.escape(n)}\b", _norm(sent)) and \
                    sent not in out and sent not in st.source_text:
                out.append(sent)
                break
    return " ".join(out)


# --- asking the writer model ------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"stories": {"type": "array", "items": {
    "type": "object", "properties": {
        "kind": {"type": "string", "enum": [KIND_STORY, KIND_NUGGET]},
        "sentences": {"type": "array", "items": {"type": "integer"}},
        "setup": {"type": "string"}, "telling": {"type": "string"},
        "payoff": {"type": "string"}, "hook": {"type": "string"},
        "surprise": {"type": "integer"}},
    "required": ["kind", "sentences", "telling", "setup", "payoff", "hook",
                 "surprise"]}}},
    "required": ["stories"]}


def prompt(artist: str, start: int, window: list[str],
           persona: str = "", max_stories: int = 5) -> str:
    numbered = "\n".join(f"[{start + i}] {s}" for i, s in enumerate(window))
    return (
        f"You prepare material for a radio host. Below are sentences from "
        f"a biography of {artist}, numbered.\n\n{numbered}\n\n"
        "Find the parts a listener would ENJOY hearing: funny or surprising "
        "anecdotes, how they started, odd jobs, near-misses, how a name or a "
        "record came about, unusual events, collaborations. Skip chart "
        "positions, sales, awards, genre lists and praise.\n"
        "For each, write:\n"
        "- kind: 'story' when there is a real little story (something "
        "happened), 'nugget' for one interesting fact.\n"
        "- sentences: the numbers of the sentences it comes from.\n"
        "- telling: the story RETOLD in your own plain spoken words, as a "
        "person telling a friend; 1 to 3 sentences, at most 50 words. Every "
        "name, number, place and event must come from those sentences; add "
        "nothing; do not work out new numbers (no 'X years later' unless the "
        "text says it); past tense.\n"
        "- setup: optional, at most 12 words, leading in WITHOUT any fact "
        "(e.g. a question or 'here's a good one'), or empty.\n"
        "- payoff: optional, at most 12 words, a light, warm, natural "
        "reaction or punchline that adds NO fact and guesses at nobody's "
        "feelings; or empty. No philosophy, no 'isn't it', no praise.\n"
        "- hook: a few words on why a listener would enjoy it.\n"
        "- surprise: 1-5, how likely a casual listener says 'huh, I didn't "
        "know that' (5) or 'so what' (1).\n"
        + (f"The host's personality: {persona}\n" if persona else "")
        + f"Return up to {int(max_stories)}. Quality over quantity: none is fine.")


def parse(raw: Any, artist: str, sents: list[str], *, mbid: str = "",
          source_ref: str = "") -> list[Story]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return []
    out: list[Story] = []
    for s in (data.get("stories") or [])[:8]:
        if not isinstance(s, dict):
            continue
        ids = sorted({int(i) for i in (s.get("sentences") or [])
                      if isinstance(i, int) and 0 <= i < len(sents)})
        out.append(Story(
            artist=artist, kind=(s.get("kind") if s.get("kind") in
                                 (KIND_STORY, KIND_NUGGET) else KIND_NUGGET),
            telling=str(s.get("telling") or "").strip(),
            setup=str(s.get("setup") or "").strip(),
            payoff=str(s.get("payoff") or "").strip(),
            hook=str(s.get("hook") or "").strip()[:80],
            surprise=max(1, min(5, int(s.get("surprise") or 3)))
            if str(s.get("surprise") or "3").isdigit() else 3,
            sentences=ids, source_text=" ".join(sents[i] for i in ids),
            artist_mbid=mbid, source_ref=source_ref))
    return out


JUDGE_SCHEMA = {"type": "object", "properties": {
    "misrepresents": {"type": "boolean"}, "part": {"type": "string"}},
    "required": ["misrepresents", "part"]}


def judge_prompt(telling: str, source: str) -> str:
    """The refuse-only second opinion: does the retelling state something
    the source does not? Paraphrase, tone and omissions are fine."""
    return ("SOURCE:\n" + source + "\n\nRETELLING:\n" + telling + "\n\nDoes "
            "the RETELLING contradict the SOURCE, or state something as fact "
            "that the SOURCE does not say (a different person, number, place, "
            "order of events, or an event that did not happen)? Ignore wording, "
            "paraphrase, tone and anything left out. misrepresents=true only "
            "for a real factual difference; quote it.")


def judge_keeps(raw: Any) -> bool:
    """True keeps the story. An unreadable verdict keeps it too: the judge
    can only refuse, and only when it says so."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return True
    return not bool(data.get("misrepresents"))


def prepare_artist(material: dict[str, Any], llm_fn: Callable[[str, dict], Any],
                   *, pack: dict[str, Any] | None = None,
                   judge_fn: Callable[[str, str], bool] | None = None,
                   persona: str = "", max_chunks: int | None = None,
                   should_stop: Callable[[], bool] | None = None,
                   start_chunk: int = 0,
                   progress: dict[str, Any] | None = None,
                   on_error: Callable[[BaseException], str] | None = None,
                   max_stories: int = 5) -> list[Story]:
    """Every candidate story for one artist, checked (accepted AND refused,
    for the forensics). `llm_fn(prompt, schema) -> raw JSON`; `judge_fn(
    telling, source) -> False` only when the telling MISREPRESENTS the
    source (a refuse-only second opinion; optional).

    Resumable and interruptible: it starts at `start_chunk`, does at most
    `max_chunks` windows, and asks `should_stop` before EVERY model call --
    the judge's too. A stop before a chunk's judging is finished leaves that
    chunk's unjudged candidates refused ("deferred_unjudged") and the chunk
    to be done again. `progress` (when given) receives `chunks` (the total),
    `next_chunk` (where the next pass starts) and `done`.

    `on_error(exc)` decides what a failed model call means: "skip" (the
    window counts as done -- the historical behaviour, and the default) or
    "stop" (the pass ends; the window is done again later). Under "stop" a
    failed JUDGE call refuses the candidate ("deferred_unjudged") instead
    of keeping it unjudged. `progress["error"]` names the failure."""
    bio = material.get("bio") or ""
    sents = sentences(bio)
    windows = chunks(sents)
    out: list[Story] = []
    seen: set[str] = set()
    nxt = max(0, int(start_chunk))
    stopped, error = False, None
    for n in range(nxt, len(windows)):
        if max_chunks is not None and n - start_chunk >= max_chunks:
            break
        if should_stop is not None and should_stop():
            stopped = True                # a spoken render outranks this
            break
        start, window = windows[n]
        try:
            raw = llm_fn(prompt(material["artist"], start, window, persona,
                                max_stories), SCHEMA)
        except Exception as exc:  # noqa: BLE001 - one failed chunk costs only it
            if on_error is not None and on_error(exc) == "stop":
                stopped, error = True, type(exc).__name__
                break                     # nothing streamed is used
            logger.warning("story prep: chunk failed", exc_info=True)
            nxt = n + 1
            continue
        for st in parse(raw, material["artist"], sents,
                        mbid=material.get("mbid", ""),
                        source_ref=material.get("uri", "")):
            key = _norm(st.telling)
            if not key or key in seen:
                continue
            seen.add(key)
            check(st, bio, pack)
            if st.ok and judge_fn is not None:
                if stopped or (should_stop is not None and should_stop()):
                    stopped = True
                    st.ok = False
                    st.refusals.append("deferred_unjudged")
                else:
                    try:
                        if not judge_fn(st.telling, st.source_text):
                            st.ok = False
                            st.refusals.append("judge_misrepresents_source")
                    except Exception as exc:  # noqa: BLE001
                        if on_error is not None and on_error(exc) == "stop":
                            stopped, error = True, type(exc).__name__
                            st.ok = False
                            st.refusals.append("deferred_unjudged")
                        else:
                            st.notes.append("judge_unavailable")
            out.append(st)
        if stopped:
            break                         # this chunk is done again later
        nxt = n + 1
    if progress is not None:
        progress.update(chunks=len(windows), next_chunk=nxt,
                        done=nxt >= len(windows), error=error)
    return out
