"""Cheap, deterministic checks on a voice-clone sample.

Magic-byte sniffing and a size bound already stand in voice_library. They
answer "is this an audio container of a plausible size" -- not "is this a
recording a clone can be built from". A 2-second cough, a 6-minute
podcast, and a file whose header says WAV but whose payload will not
decode all pass the byte checks and all waste a clone.

What this module adds is the small set of facts ffmpeg can state about a
file for free, and nothing more. There is deliberately NO quality model
here: no speaker counting, no music detection, no noise classification.
Those are judgements; these are measurements.

    DECODABLE    ffmpeg can decode the stream at all
    DURATION     8-45 s accepted, 10-30 s recommended
    SPEECH       enough of the clip is not silence
    LEVEL        the loudest moment reaches -30 dBFS (volumedetect's
                 max_volume): below that, the voice is buried in the
                 room and in the microphone's own noise
    CLIPPING     the recording does not sit pinned at full scale

The first four REFUSE; clipping only WARNS, because a hot recording
still clones -- usually worse -- and refusing one would be a judgement.
LEVEL is a peak, not a loudness score: one number ffmpeg reports, one
threshold. It is checked AFTER speech: a take below the silence floor
(-40 dB) is silence whatever its peak, and what reaches LEVEL is a
recording with a voice in it that is too far down to use.
Guidance ("about 10-30 seconds of one speaker, clean speech, no music")
belongs in the interface, not in a check: nothing here can verify it.

ANALYSIS-FAILURE POSTURE, deliberately the same as audio_sanity's and for
the same reason: if ffmpeg itself is missing or broken, every upload would
otherwise be refused and custom voices would stop working because of a
tooling problem. An unavailable analyzer is reported as unavailable and
the upload proceeds on the byte checks alone.

Silence geometry is audio_sanity.analyze's, unchanged -- it takes any
container ffmpeg can read, and a voice sample is exactly the same
question that module already answers about a rendered break.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger("musicbrain")

#: What the shelf accepts. Wider than what it recommends: a good 8-second
#: take is worth more than a refusal, and 45 s is where a clone stops
#: getting better and starts costing upload time.
MIN_DURATION_S = 8.0
MAX_DURATION_S = 45.0
#: What the interface recommends, and the ONLY place that pairing lives.
RECOMMENDED_MIN_S = 10
RECOMMENDED_MAX_S = 30
#: Under this much non-silent audio there is not enough voice to clone,
#: whatever the file's length says.
MIN_SPEECH_S = 4.0
#: Above this the clip is mostly room, not speaker.
MAX_SILENCE_FRACTION = 0.7
#: The loudest moment of a usable take. Ordinary speech into a phone or a
#: laptop microphone peaks around -12..-3 dBFS; a take whose peak is below
#: this was recorded too far away or with the level turned down.
MIN_PEAK_DB = -30.0
#: Samples sitting exactly at full scale. A handful is ordinary; tens of
#: thousands is a recording that was driven into the limiter.
CLIP_SAMPLE_COUNT = 2000

ANALYSIS_TIMEOUT_S = 60

_MAX_VOL = re.compile(r"max_volume:\s*(-?[\d.]+) dB")
_HIST_0DB = re.compile(r"histogram_0db:\s*(\d+)")


class SampleRejected(ValueError):
    """The sample cannot be used. `code` is stable; `args[0]` is the
    sentence a person reads -- no exception text, no paths, no URLs."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _volume_stats(data: bytes) -> dict[str, Any]:
    """Peak and full-scale sample count via ffmpeg's volumedetect. One
    extra decode of a <=45 s clip; raises like analyze() does."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", "pipe:0", "-af", "volumedetect",
         "-f", "null", "-"],
        input=data, capture_output=True, timeout=ANALYSIS_TIMEOUT_S)
    text = proc.stderr.decode("utf-8", "replace")
    peak = _MAX_VOL.search(text)
    if proc.returncode != 0 or not peak:
        raise RuntimeError(f"volume analysis failed: {text[-200:]!r}")
    hist = _HIST_0DB.search(text)
    return {"max_volume_db": float(peak.group(1)),
            "full_scale_samples": int(hist.group(1)) if hist else 0}


def inspect(data: bytes) -> dict[str, Any]:
    """Every measurement this module can take. Never raises.

    `analysis` is the one field a caller must read first:

        "ok"           the numbers below are real
        "undecodable"  ffmpeg RAN and could not decode this file
        "unavailable"  ffmpeg could not be run at all

    The last two look identical from the outside and mean opposite
    things -- one is a bad file, the other is a broken toolchain -- so
    they are told apart HERE, by which failure the process raised, and
    never by guessing from a message.
    """
    import audio_sanity
    try:
        stats = dict(audio_sanity.analyze(data))
    except (OSError, subprocess.SubprocessError) as exc:
        # ffmpeg is missing, unrunnable, or took too long: a tooling
        # problem, never the customer's file.
        logger.warning("voice sample analysis unavailable (upload allowed "
                       "on byte checks alone): %s", str(exc)[:150])
        return {"analysis": "unavailable"}
    except Exception as exc:  # noqa: BLE001 - ffmpeg ran and refused
        logger.info("voice sample did not decode: %s", str(exc)[:150])
        return {"analysis": "undecodable"}
    stats["analysis"] = "ok"
    try:
        stats.update(_volume_stats(data))
    except Exception as exc:  # noqa: BLE001 - the loudness half may fail alone
        logger.debug("voice sample loudness unavailable: %s", str(exc)[:150])
    return stats


def check(data: bytes) -> dict[str, Any]:
    """Measure the sample and decide. Returns the stats, with a
    `warnings` list of sentences worth showing next to the result.
    Raises SampleRejected for the refusals."""
    stats = inspect(data)
    span = f"{RECOMMENDED_MIN_S}-{RECOMMENDED_MAX_S} seconds"
    if stats.get("analysis") == "undecodable":
        raise SampleRejected(
            "undecodable",
            "That file could not be decoded as audio. Send a WAV, MP3 or "
            "FLAC recording.")
    if stats.get("analysis") != "ok":
        return {"analysis": "unavailable", "warnings": []}

    duration = float(stats.get("duration_s") or 0.0)
    audible = float(stats.get("audible_s") or 0.0)
    if duration <= 0:
        raise SampleRejected(
            "undecodable",
            "That file could not be decoded as audio. Send a WAV, MP3 or "
            "FLAC recording.")
    if duration < MIN_DURATION_S:
        raise SampleRejected(
            "too_short",
            f"That recording is {duration:.0f} seconds long. A voice needs "
            f"about {span} of speech (8 seconds is the minimum).")
    if duration > MAX_DURATION_S:
        raise SampleRejected(
            "too_long",
            f"That recording is {duration:.0f} seconds long. Trim it to "
            f"about {span} ({int(MAX_DURATION_S)} seconds is the maximum).")
    if (audible < MIN_SPEECH_S
            or float(stats.get("silence_fraction") or 0.0) > MAX_SILENCE_FRACTION):
        raise SampleRejected(
            "mostly_silence",
            f"Most of that recording is silence. Send about {span} of "
            "continuous speech.")
    peak = stats.get("max_volume_db")
    if isinstance(peak, (int, float)) and peak < MIN_PEAK_DB:
        raise SampleRejected(
            "too_quiet",
            f"That recording is very quiet (its loudest moment is "
            f"{peak:.0f} dB). Record closer to the microphone, or turn its "
            "level up.")

    warnings: list[str] = []
    if stats.get("full_scale_samples", 0) > CLIP_SAMPLE_COUNT:
        warnings.append(
            "The recording peaks at full volume and may be distorted. A "
            "quieter take usually clones better.")
    if float(stats.get("max_internal_silence_s") or 0.0) > 3.0:
        warnings.append(
            "There are long silent gaps in the recording. Continuous "
            "speech clones better than a clip with pauses in it.")
    stats["warnings"] = warnings
    return stats
