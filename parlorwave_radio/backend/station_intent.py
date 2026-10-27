#!/usr/bin/env python3
"""One station intent, whoever asks for it. PURE: no I/O, no MA, no clock.

Parlorwave Mode 2 -- the Station Programmer -- has several INPUTS and one
mechanism. The Request Line, the physical Dial and the Station tab's Test
Station all produce the same object, a StationIntent, and hand it to the
same resolver (station_resolver). None of them contains Music Assistant
logic, and the resolver knows nothing about which of them asked.

    Request Line text  -\
    Dial position -> preset_id -> saved preset -> StationIntent -> resolver
    Station tab test   -/

FIELDS.

    origin     who asked: request_line | dial | station_tab. The brief
               calls this "source"; it is `origin` here because a news
               intent needs `source` for WHICH programme it is, and one
               word must not mean two things.
    kind       what sort of station: music | news. Typed, so a future All
               News preset is `kind: news, source: all_news`, never a music
               search for the word "news".
    label      the user-facing name ("Saturday Rock"). Presentation only.
    query      music only: the station term ("90s alternative rock").
    source     news only: which programme ("all_news").
    preset_id  set when the intent came from a saved preset.

A preset's LABEL and its QUERY are independent on purpose: the Dial shows
the label and the resolver reads the query, so renaming a station never
changes what it plays and retuning what it plays never renames it.

KINDS. `music` is buildable in this release. `news` is a declared,
storable, previewable-as-unsupported kind so the model is honest about the
future without implementing it; `KINDS` is the registry a later kind joins.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

ORIGINS = ("request_line", "dial", "station_tab")

KIND_MUSIC = "music"
KIND_NEWS = "news"

#: The kind registry. `buildable` says whether the resolver can build it in
#: this release; `sources` names the programmes a sourced kind accepts.
KINDS: dict[str, dict[str, Any]] = {
    KIND_MUSIC: {"buildable": True, "needs": "query"},
    KIND_NEWS: {"buildable": False, "needs": "source",
                "sources": ("all_news",)},
}

MAX_LABEL = 40
MAX_QUERY = 120
MAX_PRESETS = 12

#: The Dial settings block's own schema version (station.dial.version).
#: Separate from the settings document's version so this block can evolve
#: without touching anyone else's section.
DIAL_VERSION = 1

DIAL_DEFAULTS: dict[str, Any] = {
    "version": DIAL_VERSION,
    # Monotonic: a deleted preset's id is never handed to a new preset, so a
    # Dial position still pointing at a deleted id can never silently start
    # playing somebody else's station.
    "next_id": 1,
    # EMPTY IS THE FACTORY STATE. No station ships preconfigured.
    "presets": [],
}

PRESET_FIELDS = ("id", "label", "kind", "query", "source", "enabled",
                 "last_test")
LAST_TEST_FIELDS = ("query", "at", "status", "strategy", "tracks",
                    "artists", "viable", "weak", "variant",
                    "builder_revision")
#: The live-station fields of a last_test record. OPTIONAL, so a record
#: saved before the Station Programmer could go live reads back exactly as
#: it was stored: `viable` is the minimum-viable-station verdict, `weak` the
#: first reason it failed ('' when it passed), `variant` the Test Station
#: draw that was approved -- a live start from the preset opens with it.
LAST_TEST_LIVE_FIELDS = ("viable", "weak", "variant", "builder_revision")
#: `builder_revision` is the station builder the test ran on
#: (station_resolver.STATION_BUILDER_REVISION). OPTIONAL: a record without
#: one was made by the builder before revisions existed, loads exactly as
#: stored, and simply counts as out of date.
_ID_RE = re.compile(r"^preset_[1-9][0-9]{0,5}$")


class IntentError(ValueError):
    """An intent or preset is invalid. The message is safe to show."""


def _clean_line(value: Any, what: str, limit: int, *,
                required: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise IntentError(f"{what} must be text")
    text = " ".join(value.split())
    if any(unicodedata.category(c).startswith("C") for c in text):
        raise IntentError(f"{what} must be plain text")
    if "<" in text or ">" in text:
        raise IntentError(f"{what} must not contain < or >")
    if required and not text:
        raise IntentError(f"{what} is required")
    if len(text) > limit:
        raise IntentError(f"{what} must be at most {limit} characters")
    return text


@dataclass(frozen=True)
class StationIntent:
    kind: str
    label: str
    origin: str
    query: str = ""
    source: str = ""
    preset_id: str | None = None

    def key(self) -> str:
        """What the station IS, independent of who asked or what it is
        called: two intents with the same key build the same station."""
        body = self.query if self.kind == KIND_MUSIC else self.source
        return f"{self.kind}:{' '.join(body.casefold().split())}"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "label": self.label,
                               "origin": self.origin}
        if self.kind == KIND_MUSIC:
            out["query"] = self.query
        else:
            out["source"] = self.source
        if self.preset_id:
            out["preset_id"] = self.preset_id
        return out


def intent_from_dict(data: Any) -> StationIntent:
    """Validate and build an intent. Raises IntentError, never anything
    else, for any malformed input."""
    if not isinstance(data, dict):
        raise IntentError("a station intent must be an object")
    unknown = set(data) - {"kind", "label", "origin", "query", "source",
                           "preset_id"}
    if unknown:
        raise IntentError("unknown station intent field: "
                          + ", ".join(sorted(unknown)))
    origin = data.get("origin")
    if origin not in ORIGINS:
        raise IntentError("origin must be one of " + ", ".join(ORIGINS))
    kind = data.get("kind", KIND_MUSIC)
    if kind not in KINDS:
        raise IntentError("kind must be one of " + ", ".join(KINDS))
    preset_id = data.get("preset_id")
    if preset_id is not None and (not isinstance(preset_id, str)
                                  or not _ID_RE.match(preset_id)):
        raise IntentError("preset_id is not a preset id")
    query = source = ""
    if KINDS[kind]["needs"] == "query":
        query = _clean_line(data.get("query"), "station term", MAX_QUERY,
                            required=True)
        if data.get("source"):
            raise IntentError("a music station takes a station term, not a "
                              "source")
    else:
        source = data.get("source")
        if source not in KINDS[kind]["sources"]:
            raise IntentError(f"a {kind} station needs a source: one of "
                              + ", ".join(KINDS[kind]["sources"]))
        if data.get("query"):
            raise IntentError(f"a {kind} station is not a search; it takes "
                              "a source, not a station term")
    label = _clean_line(data.get("label"), "label", MAX_LABEL,
                        required=False) or (query or source)[:MAX_LABEL]
    return StationIntent(kind=kind, label=label, origin=origin, query=query,
                         source=source, preset_id=preset_id)


# --- presets -------------------------------------------------------------------

def _clean_last_test(value: Any) -> dict[str, Any] | None:
    """The small record of the last successful Test Station for a preset.
    Metadata only -- never the preview itself."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(LAST_TEST_FIELDS):
        raise IntentError("last_test is malformed")
    out: dict[str, Any] = {}
    out["query"] = _clean_line(value.get("query"), "last_test.query",
                               MAX_QUERY, required=False)
    for key in ("at", "status", "strategy"):
        v = value.get(key, "")
        if not isinstance(v, str) or len(v) > 40:
            raise IntentError(f"last_test.{key} is malformed")
        out[key] = v
    for key in ("tracks", "artists"):
        v = value.get(key, 0)
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 500:
            raise IntentError(f"last_test.{key} is malformed")
        out[key] = v
    if "viable" in value:
        if not isinstance(value["viable"], bool):
            raise IntentError("last_test.viable is malformed")
        out["viable"] = value["viable"]
    if "weak" in value:
        v = value["weak"]
        if not isinstance(v, str) or len(v) > 40:
            raise IntentError("last_test.weak is malformed")
        out["weak"] = v
    if "variant" in value:
        v = value["variant"]
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 9999:
            raise IntentError("last_test.variant is malformed")
        out["variant"] = v
    if "builder_revision" in value:
        v = value["builder_revision"]
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 9999:
            raise IntentError("last_test.builder_revision is malformed")
        out["builder_revision"] = v
    return out


def clean_preset(raw: Any, *, require_id: bool) -> dict[str, Any]:
    """One preset, validated and normalised. Raises IntentError."""
    if not isinstance(raw, dict):
        raise IntentError("each preset must be an object")
    unknown = set(raw) - set(PRESET_FIELDS)
    if unknown:
        raise IntentError("unknown preset field: " + ", ".join(sorted(unknown)))
    pid = raw.get("id")
    if pid in (None, ""):
        if require_id:
            raise IntentError("a stored preset has no id")
        pid = None
    elif not isinstance(pid, str) or not _ID_RE.match(pid):
        raise IntentError("preset id is malformed")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise IntentError("preset enabled must be true or false")
    # The preset IS an intent with a stable id: validate through the one
    # intent validator so a preset can never hold something the resolver
    # would refuse.
    intent = intent_from_dict({k: raw[k] for k in ("kind", "label", "query",
                                                   "source") if k in raw}
                              | {"origin": "dial"})
    if not _clean_line(raw.get("label"), "label", MAX_LABEL, required=False):
        raise IntentError("a preset needs a dial label")
    out = {"id": pid, "label": intent.label, "kind": intent.kind,
           "query": intent.query, "source": intent.source,
           "enabled": enabled}
    last = _clean_last_test(raw.get("last_test"))
    if last is not None:
        out["last_test"] = last
    return out


def _id_number(pid: str) -> int:
    return int(pid.split("_", 1)[1])


def validate_dial(block: Any) -> dict[str, Any]:
    """A STORED station.dial block, validated whole. Raises IntentError."""
    if not isinstance(block, dict):
        raise IntentError("station.dial must be an object")
    unknown = set(block) - set(DIAL_DEFAULTS)
    if unknown:
        raise IntentError("unknown station.dial field: "
                          + ", ".join(sorted(unknown)))
    if block.get("version") != DIAL_VERSION:
        raise IntentError("station.dial was written by a different version")
    next_id = block.get("next_id")
    if not isinstance(next_id, int) or isinstance(next_id, bool) \
            or not 1 <= next_id <= 999999:
        raise IntentError("station.dial.next_id is malformed")
    presets = block.get("presets")
    if not isinstance(presets, list) or len(presets) > MAX_PRESETS:
        raise IntentError(f"station.dial.presets must be a list of at most "
                          f"{MAX_PRESETS}")
    cleaned = [clean_preset(p, require_id=True) for p in presets]
    ids = [p["id"] for p in cleaned]
    if len(set(ids)) != len(ids):
        raise IntentError("two presets share an id")
    if any(_id_number(i) >= next_id for i in ids):
        raise IntentError("a preset id is ahead of station.dial.next_id")
    return {"version": DIAL_VERSION, "next_id": next_id, "presets": cleaned}


def apply_presets(stored: dict[str, Any], incoming: Any) -> dict[str, Any]:
    """The Dial block after a person saves the preset list.

    The list REPLACES the stored one (removing a preset is omitting it).
    A preset without an id is new and is minted one here; a preset WITH an
    id must name one that exists -- ids are the backend's to issue, never
    the browser's, so a Dial mapping can trust them. Raises IntentError."""
    if not isinstance(incoming, list):
        raise IntentError("presets must be a list")
    if len(incoming) > MAX_PRESETS:
        raise IntentError(f"at most {MAX_PRESETS} dial presets")
    base = validate_dial(stored)
    existing = {p["id"] for p in base["presets"]}
    next_id = base["next_id"]
    out = []
    for raw in incoming:
        preset = clean_preset(raw, require_id=False)
        if preset["id"] is None:
            preset["id"] = f"preset_{next_id}"
            next_id += 1
        elif preset["id"] not in existing:
            raise IntentError(f"unknown preset id: {preset['id']}")
        out.append(preset)
    ids = [p["id"] for p in out]
    if len(set(ids)) != len(ids):
        raise IntentError("two presets share an id")
    return validate_dial({"version": DIAL_VERSION, "next_id": next_id,
                          "presets": out})


def intent_for_preset(preset: dict[str, Any], *, origin: str = "dial"
                      ) -> StationIntent:
    """The intent a saved preset stands for. This is the whole of what the
    Dial will ever do: position -> preset_id -> this -> resolver."""
    data = {"kind": preset.get("kind"), "label": preset.get("label"),
            "origin": origin, "preset_id": preset.get("id")}
    if preset.get("kind") == KIND_MUSIC:
        data["query"] = preset.get("query")
    else:
        data["source"] = preset.get("source")
    return intent_from_dict(data)
