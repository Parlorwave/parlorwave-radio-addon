"""The Voice Library: recordings kept on this
installation, apart from the three Custom slots.

The slots (voice_library.CUSTOM_SLOTS) stay exactly what they were: the
voice engine's store holds the recording each one speaks with, under a
fixed asset name. This module adds a SECOND, local copy of recordings the
user wants to keep, so a slot is an assignment rather than the only home
of a recording:

    <base_dir>/voice_collection/index.json      names, identities, state
    <base_dir>/voice_collection/<identity>.<ext> the recording's own bytes

One entry per recording, keyed by its identity (the whole SHA-256 of its
bytes, voice_library.sample_identity) -- importing the same file twice
finds the entry it already has.

KEEPING A SLOT'S RECORDING (keep_slot). A slot's current recording is
copied in from the engine's own download route, read-only, and the copy
is proven against what the engine lists for it (its MD5 and size) before
it is filed. Nothing about the slot changes: not its name, identity,
acknowledgement, scope or selection. It runs:

  * when the library is listed (best effort: a slot already kept under
    the same engine stamp is not downloaded again), and
  * before any slot's recording is replaced or removed (strict: if the
    copy cannot be made and proven, the replace or removal is refused and
    the slot keeps what it had). So reassigning a slot never costs the
    recording it held.

ASSIGNING (assign) puts an entry's recording into a slot through the same
staged swap a Booth replace uses (voice_library._put_sample), with a fresh
acknowledgement. The entry stays in the library.

DELETING is the only way a recording leaves the library, and it is hard
to do by accident: the entry must be archived first, must not be the
recording any slot holds, and the request must repeat the entry's name.

Nothing here reaches Music Assistant, a queue, a break or an announcement,
and no byte of audio is logged.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import voice_library as vl

logger = logging.getLogger("musicbrain")

DIR_NAME = "voice_collection"
INDEX_NAME = "index.json"
SCHEMA = 1
#: Formats a kept recording may be stored as: what the Booth accepts, plus
#: what the engine itself accepts for a slot's recording.
FORMATS = ("wav", "mp3", "flac", "m4a", "ogg")
ORIGIN_IMPORT = "import"
ORIGIN_SLOT = "slot"
MAX_ENTRIES = 50

_LOCK = threading.RLock()

KEEP_FAILED_MESSAGE = (
    "{name}'s current recording could not be copied into your Voice "
    "Library, so it was left exactly as it was. Check that the voice "
    "engine is running, then try again.")


class VoiceCollectionError(vl.VoiceError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class VoiceInUse(VoiceCollectionError):
    pass


# --- the store ---------------------------------------------------------------

def _dir() -> Path:
    import config
    return Path(config.current().base_dir) / DIR_NAME


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "entries": {}, "slots": {}}


def _load() -> dict[str, Any]:
    """The index, or an empty one when none exists yet. An index that
    exists and cannot be read is an ERROR, never an empty library: writing
    over it would forget every kept recording."""
    p = _dir() / INDEX_NAME
    if not p.exists():
        return _empty()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("voice library index unreadable: %s", type(exc).__name__)
        raise VoiceCollectionError(
            "library_unreadable",
            "The Voice Library could not be read, so nothing in it was "
            "changed.") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), dict):
        raise VoiceCollectionError(
            "library_unreadable",
            "The Voice Library could not be read, so nothing in it was "
            "changed.")
    doc.setdefault("slots", {})
    return doc


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _save(doc: dict[str, Any]) -> None:
    _write_atomic(_dir() / INDEX_NAME,
                  json.dumps(doc, indent=1, sort_keys=True).encode("utf-8"))


def _audio_path(entry: dict[str, Any]) -> Path:
    return _dir() / f"{entry['identity']}.{entry['format']}"


def _read_audio(entry: dict[str, Any]) -> bytes:
    """An entry's recording, proven to be the one the entry names."""
    try:
        data = _audio_path(entry).read_bytes()
    except OSError as exc:
        raise VoiceCollectionError(
            "recording_missing",
            "That recording is missing from the Voice Library.") from exc
    if vl.sample_identity(data) != entry["identity"]:
        raise VoiceCollectionError(
            "recording_damaged",
            "That recording in the Voice Library is damaged.")
    return data


def _entry_id(identity: str) -> str:
    return identity[:16]


def _find(doc: dict[str, Any], entry_id: str) -> dict[str, Any]:
    entry = doc["entries"].get(str(entry_id or ""))
    if not entry:
        raise VoiceCollectionError("unknown_recording",
                                   "There is no such recording in the "
                                   "Voice Library.")
    return entry


def _clean_name(name: Any) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise VoiceCollectionError("name_required", "Give the recording a name.")
    if len(name) > vl.MAX_NAME_LEN:
        raise VoiceCollectionError(
            "name_too_long",
            f"A name can be at most {vl.MAX_NAME_LEN} characters.")
    return name


def _name_taken(doc: dict[str, Any], name: str, except_id: str = "") -> bool:
    return any(e["name"].casefold() == name.casefold()
               for eid, e in doc["entries"].items() if eid != except_id)


def _free_name(doc: dict[str, Any], name: str) -> str:
    """A kept slot recording takes the slot's name, numbered when the
    library already has one by that name (a slot re-recorded twice)."""
    base, n = name[:vl.MAX_NAME_LEN - 4], 2
    while _name_taken(doc, name):
        name, n = f"{base} ({n})", n + 1
    return name


def _add(doc: dict[str, Any], data: bytes, kind: str, name: str,
         origin: str, authorization: dict[str, Any],
         origin_slot: str = "") -> dict[str, Any]:
    """File a recording (audio first, then the index). Returns the entry
    already holding these exact bytes when there is one."""
    identity = vl.sample_identity(data)
    eid = _entry_id(identity)
    if eid in doc["entries"]:
        return doc["entries"][eid]
    if len(doc["entries"]) >= MAX_ENTRIES:
        raise VoiceCollectionError(
            "library_full",
            f"The Voice Library holds {MAX_ENTRIES} recordings. Delete one "
            "you no longer need first.")
    entry = {"id": eid, "identity": identity, "name": name, "format": kind,
             "bytes": len(data), "origin": origin, "origin_slot": origin_slot,
             "authorization": authorization, "created_at": _now(),
             "archived": False}
    _write_atomic(_audio_path(entry), data)
    if vl.sample_identity(_audio_path(entry).read_bytes()) != identity:
        raise VoiceCollectionError("recording_damaged",
                                   "The recording could not be saved intact.")
    doc["entries"][eid] = entry
    return entry


# --- which recording each slot holds ------------------------------------------

def _slot_identities(doc: dict[str, Any]) -> dict[str, str]:
    """The identity of the recording each slot holds, as best this
    installation knows it: the slot's own fingerprint when it has one,
    else the one this library proved when it kept the slot's recording."""
    slots = vl._slots()
    out: dict[str, str] = {}
    for slot in vl.CUSTOM_SLOTS:
        ident = str((slots.get(slot) or {}).get("identity") or "")
        if not ident:
            ident = str((doc["slots"].get(slot) or {}).get("identity") or "")
        if ident:
            out[slot] = ident
    return out


def _slots_holding(doc: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    held = _slot_identities(doc)
    return [s for s in vl.CUSTOM_SLOTS if held.get(s) == entry["identity"]]


def _stamp(rec: dict[str, Any]) -> str:
    return "|".join(str(rec.get(k) or "") for k in
                    ("file_hash", "file_size", "upload_date"))


def keep_slot(slot: str, *, records_fn=None, download_fn=None
              ) -> dict[str, Any] | None:
    """Copy the slot's current recording into the library, proven, unless
    it is already there under the same engine stamp. Returns the entry, or
    None when the engine lists no recording for the slot. Raises
    VoiceCollectionError when the copy cannot be made or proven -- the
    caller decides whether that stops it. Never changes the slot."""
    import providers
    if slot not in vl.CUSTOM_SLOTS:
        raise vl.VoiceError(f"unknown custom slot: {slot}")
    asset = vl.CUSTOM_ASSET[slot]
    name = vl._custom_name(slot, vl._slots().get(slot) or {})
    failed = VoiceCollectionError("keep_failed",
                                  KEEP_FAILED_MESSAGE.format(name=name))
    try:
        records = (records_fn or providers.chatterbox_voices)()
    except Exception as exc:  # noqa: BLE001 - the engine boundary
        raise failed from exc
    rec = next((r for r in records if r.get("name") == asset), None)
    if rec is None:
        return None
    stamp = _stamp(rec)
    with _LOCK:
        doc = _load()
        seen = doc["slots"].get(slot) or {}
        eid = _entry_id(str(seen.get("identity") or "")) if seen else ""
        if seen.get("stamp") == stamp and eid in doc["entries"] \
                and _audio_path(doc["entries"][eid]).exists():
            return doc["entries"][eid]
    try:
        data = (download_fn or providers.chatterbox_download_voice)(asset)
    except Exception as exc:  # noqa: BLE001 - the engine boundary
        logger.warning("voice library could not download %s: %s", slot,
                       str(exc)[:160])
        raise failed from exc
    # proven against the engine's own listing, where it gives one
    want_md5 = str(rec.get("file_hash") or "")
    want_size = rec.get("file_size")
    if not data or (want_md5 and hashlib.md5(data).hexdigest() != want_md5) \
            or (isinstance(want_size, int) and want_size != len(data)):
        logger.warning("voice library: the copy of %s did not match the "
                       "engine's listing", slot)
        raise failed
    try:
        kind = vl._sniff(data)
    except vl.VoiceError:
        kind = str(rec.get("file_extension") or "").lstrip(".").lower()
        if kind not in FORMATS:
            raise failed
    with _LOCK:
        doc = _load()
        try:
            entry = _add(doc, data, kind, _free_name(doc, name), ORIGIN_SLOT,
                         vl.authorization_of(slot), origin_slot=slot)
        except OSError as exc:
            logger.warning("voice library could not save %s: %s", slot,
                           type(exc).__name__)
            raise failed from exc
        doc["slots"][slot] = {"stamp": stamp, "identity": entry["identity"],
                              "kept_at": _now()}
        _save(doc)
    logger.info("voice library kept the recording of %s", slot)
    return entry


def forget_slot(slot: str) -> None:
    """The slot's recording was removed: the library's note of which
    recording it held is dropped, so the kept copy is no longer counted as
    in use. The copy itself stays. Best effort; never raises."""
    try:
        with _LOCK:
            if not (_dir() / INDEX_NAME).exists():
                return
            doc = _load()
            if doc["slots"].pop(slot, None) is not None:
                _save(doc)
    except Exception:  # noqa: BLE001 - the slot is already gone
        logger.warning("voice library could not note %s as empty", slot)


def keep_all(*, records_fn=None, download_fn=None) -> list[str]:
    """Best effort over every slot; returns what could not be kept.

    With no voice engine configured there is no slot recording to keep, so
    nothing is reported: a fresh installation's Saved recordings must not
    say a recording "could not be copied" when there is neither engine nor
    recording. keep_slot itself stays fail-closed for a replace or delete."""
    if records_fn is None:
        import providers
        if not providers.chatterbox_voices_url():
            return []
    problems = []
    for slot in vl.CUSTOM_SLOTS:
        try:
            keep_slot(slot, records_fn=records_fn, download_fn=download_fn)
        except vl.VoiceError as exc:
            problems.append(str(exc))
    return problems


# --- the library's own actions -------------------------------------------------

def _view(doc: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    held = _slots_holding(doc, entry)
    slots = vl._slots()
    return {"id": entry["id"], "name": entry["name"],
            "format": entry["format"], "bytes": entry["bytes"],
            "origin": entry["origin"], "created_at": entry["created_at"],
            "archived": bool(entry.get("archived")),
            "permission": (entry.get("authorization") or {}).get("status")
            == vl.AUTHORIZATION_GRANTED,
            "in_slots": [_slot_view(s, slots) for s in held]}


def _slot_view(slot: str, slots: dict[str, Any]) -> dict[str, str]:
    """A slot as a person reads it: `label` is always "Custom N" (the
    Booth never shows a storage id), `name` the voice's own name."""
    return {"slot": slot, "label": slot.replace("_", " ").title(),
            "name": vl._custom_name(slot, slots.get(slot) or {})}


def listing(*, keep: bool = True, records_fn=None, download_fn=None
            ) -> dict[str, Any]:
    """Every kept recording, newest first, and which slot holds which.
    Keeps the slots' current recordings first, best effort."""
    problems = keep_all(records_fn=records_fn, download_fn=download_fn) \
        if keep else []
    with _LOCK:
        doc = _load()
    entries = sorted(doc["entries"].values(),
                     key=lambda e: e["created_at"], reverse=True)
    slots = vl._slots()
    return {"entries": [_view(doc, e) for e in entries],
            "slots": [_slot_view(s, slots) for s in vl.CUSTOM_SLOTS],
            "problems": problems, "max_entries": MAX_ENTRIES}


def import_recording(name: Any, data: bytes, *, authorized: Any = False,
                     consent_version: Any = "", source: Any = "file",
                     now_fn=None) -> dict[str, Any]:
    """Add a recording to the library. Same acknowledgement and the same
    audio checks as making a voice in the Booth; nothing reaches the
    voice engine and no slot changes."""
    vl._require_authorization(authorized, consent_version)
    source = vl._check_source(source)
    name = _clean_name(name)
    kind, stats = vl._validate_sample(data)
    with _LOCK:
        doc = _load()
        eid = _entry_id(vl.sample_identity(data))
        if eid in doc["entries"]:
            return {"created": False, **_view(doc, doc["entries"][eid])}
        if _name_taken(doc, name):
            raise VoiceCollectionError(
                "name_taken", f'Another recording is already called "{name}".')
        entry = _add(doc, data, kind, name, ORIGIN_IMPORT,
                     vl._authorization_record(source, now_fn))
        _save(doc)
        logger.info("voice library: recording imported (%s, %s)", kind, source)
        return {"created": True, "seconds": stats.get("duration_s"),
                "warnings": stats.get("warnings") or [], **_view(doc, entry)}


def rename(entry_id: str, name: Any) -> dict[str, Any]:
    name = _clean_name(name)
    with _LOCK:
        doc = _load()
        entry = _find(doc, entry_id)
        if _name_taken(doc, name, entry["id"]):
            raise VoiceCollectionError(
                "name_taken", f'Another recording is already called "{name}".')
        entry["name"] = name
        _save(doc)
        return _view(doc, entry)


def set_archived(entry_id: str, archived: Any) -> dict[str, Any]:
    if not isinstance(archived, bool):
        raise VoiceCollectionError("bad_request", "Archive is on or off.")
    with _LOCK:
        doc = _load()
        entry = _find(doc, entry_id)
        if archived and _slots_holding(doc, entry):
            raise VoiceInUse("recording_in_use",
                             "A recording a voice is using cannot be "
                             "archived. Assign that voice another "
                             "recording first.")
        entry["archived"] = archived
        _save(doc)
        return _view(doc, entry)


def audio(entry_id: str) -> tuple[bytes, str]:
    """The recording itself, for this browser to play. Previewing an entry
    is hearing what was recorded: nothing is synthesized and the engine is
    not asked."""
    with _LOCK:
        entry = _find(_load(), entry_id)
    ctype = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
             "m4a": "audio/mp4", "ogg": "audio/ogg"}[entry["format"]]
    return _read_audio(entry), ctype


def delete(entry_id: str, confirm: Any) -> dict[str, Any]:
    """Remove a recording for good. Only an ARCHIVED entry that no slot
    holds, and only when `confirm` repeats its name exactly."""
    with _LOCK:
        doc = _load()
        entry = _find(doc, entry_id)
        if _slots_holding(doc, entry):
            raise VoiceInUse("recording_in_use",
                             "A voice is using this recording, so it cannot "
                             "be deleted.")
        if not entry.get("archived"):
            raise VoiceCollectionError(
                "not_archived", "Archive the recording before deleting it.")
        if str(confirm or "") != entry["name"]:
            raise VoiceCollectionError(
                "not_confirmed",
                "Deleting needs the recording's name, to be sure.")
        del doc["entries"][entry["id"]]
        for slot, seen in list(doc["slots"].items()):
            if (seen or {}).get("identity") == entry["identity"]:
                del doc["slots"][slot]
        _save(doc)
        try:
            _audio_path(entry).unlink()
        except FileNotFoundError:
            pass
        logger.info("voice library: a recording was deleted")
        return {"id": entry["id"], "deleted": True}


def assign(entry_id: str, slot: str, *, authorized: Any = False,
           consent_version: Any = "", records_fn=None, download_fn=None,
           **engine) -> dict[str, Any]:
    """Put an entry's recording into a Custom slot. The slot's previous
    recording is kept in the library first (strict, inside the swap), and
    the entry stays here. A fresh acknowledgement is required, as for any
    recording that becomes a voice. The slot takes the entry's name when
    no other voice has it."""
    vl._require_authorization(authorized, consent_version)
    if slot not in vl.CUSTOM_SLOTS:
        raise vl.VoiceError("Choose Custom 1, 2 or 3.")
    with _LOCK:
        doc = _load()
        entry = _find(doc, entry_id)
    if entry.get("archived"):
        raise VoiceCollectionError("archived",
                                   "Restore the recording before assigning it.")
    data = _read_audio(entry)
    if _slot_identities(doc).get(slot) == entry["identity"]:
        return {"slot": slot, "unchanged": True, **_view(doc, entry)}
    source = (entry.get("authorization") or {}).get("source")
    result = vl._put_sample(
        slot, data, entry["format"], authorized=authorized,
        consent_version=consent_version,
        source=source if source in vl.SAMPLE_SOURCES else "file",
        keep_fn=lambda s: keep_slot(s, records_fn=records_fn,
                                    download_fn=download_fn),
        **engine)
    renamed = False
    if vl._slots().get(slot, {}).get("name") != entry["name"]:
        try:
            vl.rename(slot, entry["name"])
            renamed = True
        except vl.VoiceError:
            pass        # another voice has that name; the slot keeps its own
    with _LOCK:
        doc = _load()
        doc["slots"].pop(slot, None)    # the slot's fingerprint now speaks
        _save(doc)
        view = _view(doc, doc["entries"][entry["id"]])
    logger.info("voice library: a recording was assigned to %s", slot)
    return {"slot": slot, "unchanged": False, "renamed": renamed,
            "seconds": result.get("seconds"), **view}
