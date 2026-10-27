"""Probing and tagging rendered audio, without a GPL library in-process.

WHY THIS MODULE EXISTS. Duration probing and ID3 tagging were done with
`mutagen`, which is GPL-2.0-or-later. Importing it makes the running
service a combined work with a copyleft library, and puts GPL software
inside every published image -- neither of which an Apache-2.0 product can
represent honestly. The work itself is small and the tools to do it were
already installed, so the dependency went rather than the licence.

TWO DIFFERENT BOUNDARIES, FOR TWO DIFFERENT REASONS.

READING is done by `ffprobe`, over the process boundary. ffmpeg is already
a hard runtime requirement -- it transcodes speech, pads the news intro and
analyses every break for silence -- and ffprobe ships in the same Debian
package, so this adds no dependency at all. Parity with the previous
implementation was measured rather than assumed, on the exact artifacts
this product makes and eats: a rendered break (mono 24 kHz 64 kbps), a
publisher bulletin (stereo 44.1 kHz 128 kbps CBR, no Xing), that same
bulletin after raw silence frames are appended to it, one carrying a Xing
header, the suite's synthetic frame fixture, and three kinds of rubbish.
Every duration matched to six decimal places, every invalid input failed on
both, and -- the property the news tail depends on -- the duration still
GREW when frames were appended.

WRITING is done here, in Python, and deliberately NOT by ffmpeg. ffmpeg
cannot add a tag without remuxing, which rewrites the file; its mp3 muxer
also writes a Xing header by default, and a bulletin that acquires one is a
bulletin `append_tail_silence` then refuses to pad. The promise that a
publisher's audio -- sponsorship included -- survives byte for byte is
worth more than the few dozen lines below. An ID3v2 tag is a length-
prefixed block bolted to the FRONT of the file, so writing one is: build
the block, skip any block already there, put the audio back untouched.

WHAT THIS IS NOT. It is not a tag library and must not grow into one. It
reads a duration and three stream facts, and it writes three text frames
-- plus, for program_loudness, one user-defined text frame (TXXX) that it
can also read back. Anything more belongs to whatever needs it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("musicbrain")

#: Long enough for a slow disk and a long bulletin, short enough that a
#: wedged binary cannot hold a render open. The work is metadata only.
PROBE_TIMEOUT_S = 30

#: Said once per process, not once per break: a host with no ffprobe will
#: hit this on every render, and a repeated line teaches people to skip it.
_warned_missing = False


def _ffprobe(path: Path, entries: str) -> dict[str, Any] | None:
    """One bounded ffprobe call. argv form only, never shell=True."""
    global _warned_missing
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-print_format", "json", "-show_entries", entries, str(path)],
            capture_output=True, timeout=PROBE_TIMEOUT_S)
    except FileNotFoundError:
        if not _warned_missing:
            _warned_missing = True
            logger.warning(
                "ffprobe is not installed on this host, so rendered audio "
                "cannot be validated and every break will fail as "
                "unprobeable. Install ffmpeg (which provides ffprobe); the "
                "add-on image already has it.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out on %s", path)
        return None
    if proc.returncode != 0:
        return None                     # not audio, truncated, or empty
    try:
        return json.loads(proc.stdout or b"{}")
    except ValueError:
        return None


def _first_stream(doc: dict[str, Any]) -> dict[str, Any]:
    streams = doc.get("streams") or [{}]
    return streams[0] if streams else {}


def duration(path: Path | str) -> float | None:
    """Finite-duration probe: the gate Music Assistant's prefetch relies on.

    None means "this is not a probeable finite MP3" -- for any reason, the
    same contract the callers have always had. They turn that into a
    refused break or a refused bulletin, which is the only safe reading."""
    doc = _ffprobe(Path(path), "stream=duration:format=duration")
    if doc is None:
        return None
    value = _first_stream(doc).get("duration") \
        or (doc.get("format") or {}).get("duration")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def stream_info(path: Path | str) -> dict[str, Any] | None:
    """(sample_rate, channels, bitrate, length) of an MP3, or None.

    `bitrate` comes from the STREAM and not from the container. For a CBR
    file the stream reports the encoded rate exactly (128000), while the
    container reports a measured average a hair off it (127999) -- and the
    caller divides this by 1000 to pick an encoder setting, where 127999
    would floor to 127 kbps and produce silence frames that do not match
    the bulletin they are appended to."""
    doc = _ffprobe(Path(path),
                   "stream=sample_rate,channels,bit_rate,duration:"
                   "format=duration,bit_rate")
    if doc is None:
        return None
    stream, fmt = _first_stream(doc), doc.get("format") or {}

    def number(value, cast):
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    length = number(stream.get("duration"), float) \
        or number(fmt.get("duration"), float)
    if not length or length <= 0:
        return None
    return {
        "sample_rate": number(stream.get("sample_rate"), int),
        "channels": number(stream.get("channels"), int),
        "bitrate": number(stream.get("bit_rate"), int)
                   or number(fmt.get("bit_rate"), int),
        "length": float(length),
    }


# --- ID3v2.4 text frames ------------------------------------------------------
#
# The whole of the format this needs, and none of the rest of it:
#
#   header   "ID3" | 0x04 0x00 | flags 0x00 | 4-byte synchsafe body size
#   frame    4-char id | 4-byte synchsafe size | 2 flag bytes
#            | 0x03 (UTF-8) | text | 0x00
#
# Synchsafe means seven bits per byte, so a size can never be mistaken for
# an MPEG frame sync. Everything after the header is padding, and padding
# is zeroes.

_ID3_HEADER_LEN = 10
_ENCODING_UTF8 = b"\x03"

#: A little slack at the end of the tag, so a later writer that wants a
#: longer title has somewhere to put it without moving the audio. Mutagen
#: wrote about a kilobyte; this is the same idea, smaller.
_PADDING = 256

#: What the product tags, and the only frames this writes.
FRAME_IDS = {"title": b"TIT2", "artist": b"TPE1", "album": b"TALB"}


def _synchsafe(size: int) -> bytes:
    if not 0 <= size < (1 << 28):
        raise ValueError(f"size {size} does not fit a synchsafe integer")
    return bytes(((size >> shift) & 0x7F) for shift in (21, 14, 7, 0))


def _read_synchsafe(raw: bytes) -> int:
    return ((raw[0] & 0x7F) << 21 | (raw[1] & 0x7F) << 14
            | (raw[2] & 0x7F) << 7 | (raw[3] & 0x7F))


def _text_frame(frame_id: bytes, text: str) -> bytes:
    body = _ENCODING_UTF8 + text.encode("utf-8") + b"\x00"
    return frame_id + _synchsafe(len(body)) + b"\x00\x00" + body


def existing_tag_length(head: bytes) -> int:
    """Bytes of ID3v2 tag at the front of `head`, or 0 if there is none.

    Only ever asked of the front of a file: an ID3v1 tag lives at the END,
    is not written here, and is left alone wherever it is found."""
    if len(head) < _ID3_HEADER_LEN or head[:3] != b"ID3":
        return 0
    size = _read_synchsafe(head[6:10])
    footer = _ID3_HEADER_LEN if head[5] & 0x10 else 0      # rare, but legal
    return _ID3_HEADER_LEN + size + footer


#: The user-defined text frame: 0x03 (UTF-8) | description | 0x00 | value.
USER_TEXT_ID = b"TXXX"
#: Enough of a file's front to hold any tag this module writes.
MAX_TAG_READ = 8192


def _user_text_frame(desc: str, value: str) -> bytes:
    body = _ENCODING_UTF8 + desc.encode("utf-8") + b"\x00" + value.encode("utf-8")
    return USER_TEXT_ID + _synchsafe(len(body)) + b"\x00\x00" + body


def build_tag(user_text: dict[str, str] | None = None, **frames: str) -> bytes:
    """An ID3v2.4.0 tag carrying the named text frames, padded. `user_text`
    adds one TXXX frame per (description, value)."""
    body = b"".join(_text_frame(FRAME_IDS[name], value)
                    for name, value in frames.items()
                    if name in FRAME_IDS and value)
    body += b"".join(_user_text_frame(desc, value)
                     for desc, value in (user_text or {}).items() if value)
    body += b"\x00" * _PADDING
    return b"ID3\x04\x00\x00" + _synchsafe(len(body)) + body


def tagged_bytes(audio: bytes, user_text: dict[str, str] | None = None,
                 **frames: str) -> bytes:
    """`audio` carrying exactly one ID3v2 tag: ours. Any tag already at the
    front is replaced; every audio byte after it is kept as it is.

    For a producer that has the whole artifact in memory (a rendered
    break) this is the write_text_tags result without the read-modify-
    rewrite: the file is written once, tagged, rather than written and
    then rewritten to be tagged."""
    return (build_tag(user_text, **frames)
            + audio[existing_tag_length(audio[:_ID3_HEADER_LEN]):])


def read_user_text(head: bytes, desc: str) -> str | None:
    """The value of the TXXX frame named `desc` in the ID3v2.4 tag at the
    front of `head`, or None. Reads only the tags this module writes
    (v2.4, UTF-8, no extended header); anything else answers None."""
    if len(head) < _ID3_HEADER_LEN or head[:4] != b"ID3\x04" or head[5] & 0x40:
        return None
    end = min(len(head), existing_tag_length(head[:_ID3_HEADER_LEN]))
    pos, want = _ID3_HEADER_LEN, desc.encode("utf-8") + b"\x00"
    while pos + 10 <= end and head[pos] != 0:
        size = _read_synchsafe(head[pos + 4:pos + 8])
        body = head[pos + 10:pos + 10 + size]
        if (head[pos:pos + 4] == USER_TEXT_ID and body[:1] == _ENCODING_UTF8
                and body[1:].startswith(want)):
            return body[1 + len(want):].rstrip(b"\x00").decode("utf-8", "replace")
        pos += 10 + size
    return None


class TagStripper:
    """Drops an ID3v2 block from the FRONT of a stream, chunk by chunk.

    For a producer that streams an artifact to disk (a publisher bulletin)
    and wants to write its own tag first: feed every chunk through here
    and the publisher's own leading tag -- if any -- never reaches the
    file, whatever the chunk boundaries. Call flush() at the end; a
    payload shorter than an ID3 header is passed through untouched."""

    def __init__(self, write):
        self._write = write
        self._head = b""
        self._skip: int | None = None       # None: header not yet decided

    def __call__(self, chunk: bytes) -> None:
        if self._skip is None:
            self._head += chunk
            if len(self._head) < _ID3_HEADER_LEN:
                return
            self._skip = existing_tag_length(self._head[:_ID3_HEADER_LEN])
            chunk, self._head = self._head, b""
        if self._skip:
            drop = min(self._skip, len(chunk))
            self._skip -= drop
            chunk = chunk[drop:]
        if chunk:
            self._write(chunk)

    def flush(self) -> None:
        if self._skip is None and self._head:
            self._write(self._head)
            self._head = b""


def write_text_tags(path: Path | str, **frames: str) -> bool:
    """Replace the file's ID3v2 tag with one carrying `frames`.

    Best effort, exactly as before: a tagging failure must never cost a
    perfectly good break or bulletin, so this returns False and says so in
    the log rather than raising. Every byte after the tag is written back
    unchanged -- the audio frames, and the Xing header inside the first of
    them, are never touched."""
    path = Path(path)
    try:
        original = path.read_bytes()
        audio = original[existing_tag_length(original[:_ID3_HEADER_LEN]):]
        path.write_bytes(build_tag(**frames) + audio)
        return True
    except Exception:                   # noqa: BLE001 - best effort by design
        logger.debug("ID3 tagging failed for %s", path, exc_info=True)
        return False
