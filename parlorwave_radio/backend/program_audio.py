"""The transport-safety contract for finite ParlorWave program audio.

Music Assistant queue items are FOREVER: an aired DJ break or news bulletin
stays in the queue's history as an addressable item long after its moment
has passed, and MA will re-request its media URL whenever history is
replayed -- a listener picking an old item, or MA's own flow-stream restart
resuming at a historical position (a flow restart that rewinds into
history and meets a run of HTTP 410s from retired break audio produces
break fragments, wrong-song state and finally an idle player).

The contract, therefore:

    A KNOWN program-audio token must never be transport-fatal.

  * live token (renderable state, file present)  -> its real audio
  * known token in any other situation           -> RETIRED_SILENCE
    (terminal state, superseded artifact, or a file that is already
    unlinked): HTTP 200, audio/mpeg, about 4 s of silence (RETIRED_SECONDS;
    why that long is below). The stale SPEECH is never replayed -- a
    rewound history plays a brief quiet gap where the DJ once spoke -- but
    the transport sees an ordinary short track.
  * unknown / malformed token                    -> honest 404, unchanged.
    Resilience is for our own history, not for probes.

SAFE IS NOT THE SAME AS ANONYMOUS. The asset used to carry only the
encoder's own TSSE frame, so a two-kilobyte answer to a break URL was
indistinguishable from a broken render, an empty file, or someone else's
audio -- it was transport-safe and it lied about what it was. It now
carries this product's own ID3 identity (`IDENTITY` below), and every
response that serves it is labelled `X-Parlorwave-Program: retired` while
real break audio is labelled `live`. The label changed nothing about the
transport contract: the status stayed 200 and the body a short ordinary
track. What changed is that anyone -- a customer reading a download, a
maintainer with curl, a test -- can now tell at a glance which of the two
they were handed.

LONG ENOUGH TO BE HANDED OVER. Short is not the same as safe.
A sub-second asset is too short: on a NON-Flow player (MA 2.10) a
retired token between two songs leaves the player idle with the
next song never started: Music Assistant gives a non-Flow player its next
item only while the current one is still playing, and only after a
debounced update, a 1 s preload poll, the next item's stream setup and a
fixed one-second delay -- 1.5 s at best, just under 3 s at worst, and a
hand-over that finds the player stopped is never retried. Flow Mode
buffers straight through, which is why Flow playback never shows
it. MA also keeps durations as int(), so anything under a second is stored
as 0, and an item of duration 0 never arms the hand-over at all. The asset
is therefore the fewest whole frames that reach MA's fixed worst case plus
an allowance for what is not fixed, and not one frame more: in Flow Mode
every frame of it is dead air.

The silence asset matches the DJ's own break format (24 kHz mono MPEG-2
Layer III at 64 kbps, the frame header dj_engine.MP3_ARGS writes for every
break it transcodes) and
ships with the code, so it survives restarts and redeploys and is
byte-identical everywhere. It is built from the constants below by
`retired_silence_bytes()` and tested equal to it, so its length is a number
in this file, never the side effect of a re-encode. If the asset is somehow
missing the callers fall back to the old 410 -- a misdeploy must degrade to
the previous behavior, never to serving arbitrary bytes.
"""

from __future__ import annotations

import math
from pathlib import Path

RETIRED_SILENCE = Path(__file__).resolve().parent / "assets" / "retired_silence.mp3"

#: Response header naming which of the two things this body is. Present on
#: every finite-program-audio answer, so "live" and "retired" are told apart
#: without reading the bytes.
PROGRAM_HEADER = "X-Parlorwave-Program"
PROGRAM_LIVE = "live"
PROGRAM_RETIRED = "retired"

#: The asset's ID3 identity, written by the SAME tagger that tags real
#: break audio (audio_probe.write_text_tags) and reading the same way in a
#: queue UI: "Retired break -- Parlorwave Radio" beside "DJ break --
#: Parlorwave Radio". `test_program_audio_resilience` rebuilds the tag from
#: these constants and compares it with the shipped bytes, so the two can
#: never drift apart.
IDENTITY = {"title": "Retired break",
            "artist": "Parlorwave Radio",
            "album": "Parlorwave Radio"}


#: Music Assistant 2.10.4's FIXED delays before a non-Flow player is handed
#: its next item, at their worst: the debounced player update that makes the
#: new item current (0.5 s, up to ~1 s), the preload task's 1 s poll, and
#: stream_feeder's call_later(1). The best case is 1.5 s.
HANDOFF_FIXED_WORST_S = 3.0
#: What is not fixed: the player's own report, the next item's stream
#: setup, the enqueue call.
HANDOFF_ALLOWANCE_S = 1.0

#: One frame of digital silence: MPEG-2 Layer III, 64 kbps, 24 kHz, mono,
#: no CRC -- byte for byte the frame header dj_engine.MP3_ARGS writes for
#: every break it transcodes -- with all-zero side info (no main data, so
#: each frame decodes on
#: its own to 576 zero samples) and zero fill to the frame's 192 bytes. No
#: Xing/Info header, so the probed length and the played length are one
#: number.
SILENT_FRAME = b"\xff\xf3\x84\xc4" + bytes(188)
FRAME_SAMPLES, SAMPLE_RATE = 576, 24_000

#: The fewest whole frames that reach the floor: 167 frames, 4.008 s. It is
#: not an exact second, so MA's int() keeps 8 ms of headroom above 4.
RETIRED_FRAMES = math.ceil((HANDOFF_FIXED_WORST_S + HANDOFF_ALLOWANCE_S)
                           * SAMPLE_RATE / FRAME_SAMPLES)
RETIRED_SECONDS = RETIRED_FRAMES * FRAME_SAMPLES / SAMPLE_RATE


def identity_tag() -> bytes:
    """The ID3v2.4 tag the retired-silence asset carries."""
    import audio_probe
    return audio_probe.build_tag(**IDENTITY)


def retired_silence_bytes() -> bytes:
    """The retired-silence asset, built from the constants above. The
    shipped file is tested equal to this; regenerating it is
    `RETIRED_SILENCE.write_bytes(retired_silence_bytes())`."""
    return identity_tag() + SILENT_FRAME * RETIRED_FRAMES


def tombstone() -> Path | None:
    """The silence asset, or None when it is missing (callers then answer
    410, the answer before the silence asset existed)."""
    return RETIRED_SILENCE if RETIRED_SILENCE.exists() else None


def is_tombstone(path: Path | None) -> bool:
    """Is this the retired-program placeholder rather than real program
    audio? The one place that knows, so no caller compares paths itself."""
    return path is not None and Path(path) == RETIRED_SILENCE
