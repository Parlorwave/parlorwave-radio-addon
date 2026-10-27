"""ParlorWave's durable DJ memory: what the station knows, has prepared,
has chosen, and has ACTUALLY put on the air. One small local SQLite file per
installation; nothing here calls a model or Music Assistant.

Five things are kept apart, because each answers a different question:

    sources    the biography passage a story was retold from (provider,
               field, reference, passage hash) -- "where did this come from?"
    stories    a prepared spoken story or short fact, checked before air
               -- "what could the DJ say?"
    selections a story CHOSEN for a break, with that break's token and its
               fate: selected -> aired | released -- "what was queued?"
    airings    AUTHORITATIVE confirmed broadcasts only: written when the
               break row itself reaches `completed` (dj_engine._transition,
               airing -> completed at the next real music track), never on
               generation, selection or queueing -- "what was heard?"
    feedback   the owner's vetoes and likes -- "may it air again?"

SELECTION (opportunity-driven, not clock-driven). At a break the memory
offers the best eligible material about the act that just played, the act
coming up, or an act from the last hour, under these rules:

  * one REST PER ACT across stories AND short facts (ACT_REST_MINUTES), and
    never the same act in consecutive breaks -- otherwise one act (say,
    Cold Orchard) can come up three times in half an hour when only stories rest;
  * a story is reusable STORY_REST_DAYS after it last aired (a short fact
    NUGGET_REST_DAYS): intentional reuse, not a one-shot list;
  * no two full stories in consecutive breaks;
  * sombre material (deaths, illness, violence) never enters everyday rotation,
    and neither does a list-shaped telling (a roll call of names: hard to
    follow by ear and where speech recognition missed most in testing);
  * vetoed stories never air; a pending selection reserves its story.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

STORY_REST_DAYS = 30
NUGGET_REST_DAYS = 10
ACT_REST_MINUTES = 90
RECENT_MINUTES = 60
STORY_MIN_GAP_BREAKS = 1
#: below this the existing path hosts instead: a dull or list-shaped item
#: (story_prep.record's quality) is worse radio than no story at all
MIN_QUALITY = 0.4
#: a selection with no outcome after this long is treated as released
SELECTION_TTL_S = 3 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     INTEGER PRIMARY KEY,
    artist        TEXT NOT NULL, artist_norm TEXT NOT NULL, artist_mbid TEXT,
    provider      TEXT NOT NULL, field TEXT, ref TEXT,
    passage       TEXT NOT NULL, passage_hash TEXT NOT NULL UNIQUE,
    support       TEXT,                          -- bio sentences behind a fuller name
    fetched_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stories (
    story_id      TEXT PRIMARY KEY,
    source_id     INTEGER REFERENCES sources(source_id),
    artist        TEXT NOT NULL, artist_norm TEXT NOT NULL,
    kind          TEXT NOT NULL,                 -- story | nugget
    setup TEXT, telling TEXT NOT NULL, payoff TEXT,
    proposed_setup TEXT, proposed_payoff TEXT,   -- the model's framing, as proposed
    sentences     TEXT, hook TEXT, surprise INTEGER, quality REAL,
    sombre        INTEGER NOT NULL DEFAULT 0, list_shaped INTEGER NOT NULL DEFAULT 0,
    checks        TEXT, model TEXT,
    status        TEXT NOT NULL DEFAULT 'ready', -- ready | vetoed | retired
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS stories_artist ON stories(artist_norm, status);
CREATE TABLE IF NOT EXISTS selections (
    selection_id  INTEGER PRIMARY KEY,
    story_id      TEXT NOT NULL, break_token TEXT NOT NULL UNIQUE,
    artist_norm   TEXT NOT NULL, kind TEXT NOT NULL, about TEXT,
    selected_at   REAL NOT NULL,
    state         TEXT NOT NULL DEFAULT 'selected', -- selected | aired | released
    resolved_at   REAL, outcome TEXT
);
CREATE INDEX IF NOT EXISTS selections_at ON selections(selected_at);
CREATE TABLE IF NOT EXISTS airings (
    story_id      TEXT NOT NULL, break_token TEXT NOT NULL UNIQUE,
    artist_norm   TEXT NOT NULL, kind TEXT NOT NULL,
    aired_at      REAL NOT NULL, evidence TEXT
);
CREATE INDEX IF NOT EXISTS airings_at ON airings(aired_at);
CREATE TABLE IF NOT EXISTS feedback (
    story_id TEXT NOT NULL, verdict TEXT NOT NULL, note TEXT, at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prep_log (
    artist_norm TEXT PRIMARY KEY, artist TEXT, bio_hash TEXT,
    prepared_at REAL, proposed INTEGER, kept INTEGER, outcome TEXT,
    next_chunk INTEGER NOT NULL DEFAULT 0,     -- where a partial pass resumes
    failures INTEGER NOT NULL DEFAULT 0        -- failed calls on that window
);
"""
#: prep_log outcomes after which an artist's biography counts as prepared
COMPLETE_OUTCOMES = ("prepared", "unchanged", "seeded", "no_biography")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()
                  .replace("’", "'")).strip()


class DJMemory:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path, timeout=10)
        self._c.row_factory = sqlite3.Row
        self._c.executescript(SCHEMA)
        cols = {r[1] for r in self._c.execute("PRAGMA table_info(prep_log)")}
        for col in ("next_chunk", "failures"):   # files from older versions
            if col not in cols:
                self._c.execute(f"ALTER TABLE prep_log ADD COLUMN {col} "
                                "INTEGER NOT NULL DEFAULT 0")
        self._c.commit()

    def close(self) -> None:
        self._c.close()

    # --- preparation ---------------------------------------------------------
    def add_story(self, record: dict[str, Any], *, model: str = "",
                  now: float | None = None) -> bool:
        """Store one ACCEPTED story_prep record with its source passage.
        False when the same telling for the same act is already here."""
        if not record.get("ok"):
            return False
        now = float(now if now is not None else time.time())
        src = record.get("source_text") or ""
        ph = record.get("passage_hash") or ""
        self._c.execute(
            "INSERT OR IGNORE INTO sources (artist, artist_norm, artist_mbid, "
            "provider, field, ref, passage, passage_hash, support, fetched_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (record["artist"], norm(record["artist"]),
             record.get("artist_mbid") or "", record.get("source_provider") or
             "music_assistant", record.get("source_field") or "",
             record.get("source_ref") or "", src, ph,
             record.get("support_text") or "", now))
        sid = self._c.execute("SELECT source_id FROM sources WHERE "
                              "passage_hash=?", (ph,)).fetchone()[0]
        if record.get("support_text"):
            # one passage, several tellings: keep every sentence any of them
            # needed for a fuller name
            row = self._c.execute("SELECT support FROM sources WHERE "
                                  "source_id=?", (sid,)).fetchone()
            have = row[0] or ""
            extra = [x for x in [record["support_text"]] if x not in have]
            if extra:
                self._c.execute("UPDATE sources SET support=? WHERE "
                                "source_id=?", (" ".join([have] + extra)
                                                .strip(), sid))
        cur = self._c.execute(
            "INSERT OR IGNORE INTO stories (story_id, source_id, artist, "
            "artist_norm, kind, setup, telling, payoff, proposed_setup, "
            "proposed_payoff, sentences, hook, surprise, quality, sombre, "
            "list_shaped, checks, model, created_at) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["story_id"], sid, record["artist"], norm(record["artist"]),
             record["kind"], record.get("setup") or "", record["telling"],
             record.get("payoff") or "", record.get("proposed_setup") or "",
             record.get("proposed_payoff") or "",
             json.dumps(record.get("sentences") or []), record.get("hook") or "",
             int(record.get("surprise") or 3),
             float(record.get("quality", 0.5)), int(bool(record.get("sombre"))),
             int(bool(record.get("list_shaped"))),
             json.dumps({"framing_dropped": record.get("framing_dropped") or [],
                         "notes": record.get("notes") or []}), model, now))
        self._c.commit()
        return cur.rowcount == 1

    def note_prepared(self, artist: str, bio_hash: str, proposed: int,
                      kept: int, outcome: str, now: float | None = None,
                      next_chunk: int = 0, failures: int = 0) -> None:
        self._c.execute(
            "INSERT OR REPLACE INTO prep_log (artist_norm, artist, bio_hash, "
            "prepared_at, proposed, kept, outcome, next_chunk, failures) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (norm(artist), artist, bio_hash, float(now or time.time()),
             proposed, kept, outcome, int(next_chunk), int(failures)))
        self._c.commit()

    def prep_state(self, artist: str) -> dict[str, Any] | None:
        row = self._c.execute("SELECT * FROM prep_log WHERE artist_norm=?",
                              (norm(artist),)).fetchone()
        return dict(row) if row else None

    def prepared_hash(self, artist: str) -> str | None:
        """The biography hash of the last COMPLETE preparation (a partial
        pass does not count)."""
        st = self.prep_state(artist)
        return st["bio_hash"] if st and st["outcome"] in COMPLETE_OUTCOMES \
            else None

    def last_prep_at(self) -> float:
        row = self._c.execute("SELECT MAX(prepared_at) FROM prep_log").fetchone()
        return float(row[0] or 0.0)

    # --- owner feedback --------------------------------------------------------
    def feedback(self, story_id: str, verdict: str, note: str = "",
                 now: float | None = None) -> bool:
        """'veto' (never again), 'retire' (done with it), 'like' (keep)."""
        if verdict not in ("veto", "retire", "like"):
            raise ValueError(verdict)
        hit = self._c.execute("SELECT 1 FROM stories WHERE story_id=?",
                              (story_id,)).fetchone()
        if not hit:
            return False
        self._c.execute("INSERT INTO feedback (story_id, verdict, note, at) "
                        "VALUES (?,?,?,?)",
                        (story_id, verdict, note[:200], float(now or time.time())))
        if verdict in ("veto", "retire"):
            status = "vetoed" if verdict == "veto" else "retired"
            self._c.execute("UPDATE stories SET status=? WHERE story_id=?",
                            (status, story_id))
        self._c.commit()
        return True

    # --- the broadcast lifecycle -----------------------------------------------
    def reserve(self, story: dict[str, Any], break_token: str,
                now: float | None = None) -> bool:
        """The story was CHOSEN for this break. Not an airing."""
        cur = self._c.execute(
            "INSERT OR IGNORE INTO selections (story_id, break_token, "
            "artist_norm, kind, about, selected_at) VALUES (?,?,?,?,?,?)",
            (story["story_id"], break_token, norm(story["artist"]),
             story["kind"], story.get("about"), float(now or time.time())))
        self._c.commit()
        return cur.rowcount == 1

    def confirm_aired(self, break_token: str, now: float | None = None,
                      evidence: str = "completed") -> bool:
        """The break row reached `completed` (airing -> completed at the next
        real music track): the only authoritative 'it was heard'."""
        row = self._c.execute("SELECT * FROM selections WHERE break_token=?",
                              (break_token,)).fetchone()
        if row is None or row["state"] != "selected":
            return False              # unknown, already aired, or released
        now = float(now or time.time())
        self._c.execute("INSERT OR IGNORE INTO airings (story_id, break_token, "
                        "artist_norm, kind, aired_at, evidence) VALUES "
                        "(?,?,?,?,?,?)", (row["story_id"], break_token,
                                          row["artist_norm"], row["kind"], now,
                                          evidence))
        self._c.execute("UPDATE selections SET state='aired', resolved_at=?, "
                        "outcome=? WHERE break_token=?",
                        (now, evidence, break_token))
        self._c.commit()
        return True

    def release(self, break_token: str, outcome: str,
                now: float | None = None) -> bool:
        """The break never aired (revoked, failed, missed): the story is
        free again, and nothing counts as heard."""
        cur = self._c.execute(
            "UPDATE selections SET state='released', resolved_at=?, outcome=? "
            "WHERE break_token=? AND state='selected'",
            (float(now or time.time()), outcome, break_token))
        self._c.commit()
        return cur.rowcount == 1

    def pending_tokens(self) -> list[str]:
        """Break tokens whose story is chosen but not yet resolved."""
        return [r[0] for r in self._c.execute(
            "SELECT break_token FROM selections WHERE state='selected'")]

    # --- reading -----------------------------------------------------------------
    def counts(self) -> dict[str, Any]:
        out = {r["kind"]: r["n"] for r in self._c.execute(
            "SELECT kind, COUNT(*) n FROM stories WHERE status='ready' AND "
            "sombre=0 AND list_shaped=0 AND quality >= ? GROUP BY kind",
            (MIN_QUALITY,))}
        out["below_quality_floor"] = self._c.execute(
            "SELECT COUNT(*) FROM stories WHERE status='ready' AND sombre=0 "
            "AND list_shaped=0 AND quality < ?", (MIN_QUALITY,)).fetchone()[0]
        out["sombre_held_back"] = self._c.execute(
            "SELECT COUNT(*) FROM stories WHERE status='ready' AND sombre=1"
        ).fetchone()[0]
        out["list_shaped_held_back"] = self._c.execute(
            "SELECT COUNT(*) FROM stories WHERE status='ready' AND sombre=0 "
            "AND list_shaped=1").fetchone()[0]
        out["artists"] = self._c.execute(
            "SELECT COUNT(DISTINCT artist_norm) FROM stories WHERE "
            "status='ready'").fetchone()[0]
        out["vetoed"] = self._c.execute(
            "SELECT COUNT(*) FROM stories WHERE status='vetoed'").fetchone()[0]
        out["aired"] = self._c.execute("SELECT COUNT(*) FROM airings"
                                       ).fetchone()[0]
        out["pending_selections"] = self._c.execute(
            "SELECT COUNT(*) FROM selections WHERE state='selected'"
        ).fetchone()[0]
        return out

    def recent_airings(self, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(r) for r in self._c.execute(
            "SELECT a.*, s.artist, s.telling FROM airings a JOIN stories s "
            "USING (story_id) ORDER BY aired_at DESC LIMIT ?", (limit,))]

    def _recent_selections(self, now: float, minutes: float
                           ) -> list[sqlite3.Row]:
        return self._c.execute(
            "SELECT * FROM selections WHERE selected_at >= ? AND state != "
            "'released' ORDER BY selected_at DESC",
            (now - minutes * 60,)).fetchall()

    def candidates(self, *, ending_artist: str = "", next_artist: str = "",
                   recent_artists: Iterable[tuple[str, float]] = (),
                   breaks_since_story: int = 99, last_break_act: str = "",
                   now: float | None = None) -> list[dict[str, Any]]:
        """Every eligible item for THIS break, best first (the caller takes
        the first that fits its spoken-time budget)."""
        now = float(now if now is not None else time.time())
        # stale pending selections no longer reserve anything
        self._c.execute("UPDATE selections SET state='released', "
                        "outcome='expired', resolved_at=? WHERE state='selected'"
                        " AND selected_at < ?", (now, now - SELECTION_TTL_S))
        self._c.commit()
        last_aired = {r[0]: r[1] for r in self._c.execute(
            "SELECT story_id, MAX(aired_at) FROM airings GROUP BY story_id")}
        reserved = {r[0] for r in self._c.execute(
            "SELECT story_id FROM selections WHERE state='selected'")}
        # ONE rest per act, across stories AND short facts, counting both
        # confirmed airings and pending selections
        rested = {r[0] for r in self._c.execute(
            "SELECT artist_norm FROM airings WHERE aired_at >= ?",
            (now - ACT_REST_MINUTES * 60,))}
        rested |= {r["artist_norm"] for r in self._recent_selections(
            now, ACT_REST_MINUTES)}
        last_act = norm(last_break_act)
        sides: list[tuple[str, str]] = []
        for artist, about in ((ending_artist, "ending"), (next_artist, "next")):
            if artist and norm(artist) not in {norm(a) for a, _ in sides}:
                sides.append((artist, about))
        for artist, at in recent_artists:
            if (now - float(at) <= RECENT_MINUTES * 60
                    and norm(artist) not in {norm(a) for a, _ in sides}):
                sides.append((artist, "earlier"))
        kinds = ["story", "nugget"] if breaks_since_story >= \
            STORY_MIN_GAP_BREAKS else ["nugget"]
        out: list[tuple[float, dict[str, Any]]] = []
        for rank, (artist, about) in enumerate(sides):
            a = norm(artist)
            if a in rested or (last_act and a == last_act):
                continue
            for row in self._c.execute(
                    "SELECT st.*, so.provider AS source_provider, so.field AS "
                    "source_field, so.ref AS source_ref, so.passage AS "
                    "source_text, so.support AS source_support FROM stories st "
                    "LEFT JOIN sources so USING (source_id) WHERE "
                    "st.artist_norm=? AND st.status='ready' AND st.sombre=0 AND "
                    "st.list_shaped=0 AND st.quality >= ?", (a, MIN_QUALITY)):
                kind = row["kind"]
                if kind not in kinds:
                    continue
                if kind == "nugget" and about == "earlier":
                    continue          # a bare fact about an old song is a non sequitur
                if row["story_id"] in reserved:
                    continue
                at = last_aired.get(row["story_id"])
                rest = STORY_REST_DAYS if kind == "story" else NUGGET_REST_DAYS
                if at is not None and now - at < rest * 86400:
                    continue
                d = dict(row)
                d["about"], d["why"] = about, f"{kind}:{about}"
                d["sentences"] = json.loads(d.get("sentences") or "[]")
                score = (float(d.get("quality") or 0.5) - 0.05 * rank
                         + (0.05 if kind == "story" else 0.0)
                         - (0.1 if at is not None else 0.0))   # fresh first
                out.append((score, d))
        out.sort(key=lambda x: (-x[0], x[1]["story_id"]))
        return [d for _, d in out]

    def source_of(self, story_id: str) -> dict[str, Any] | None:
        row = self._c.execute(
            "SELECT so.* FROM stories st JOIN sources so USING (source_id) "
            "WHERE st.story_id=?", (story_id,)).fetchone()
        return dict(row) if row else None
