"""Loudness levelling for program audio that Music Assistant plays as a
SOUND EFFECT (the prototype, playback.program_audio_as_sound_effect).

WHY. Music Assistant normalises a TRACK toward its loudness target (observed
in testing: -14 LUFS) but turns normalisation off for a sound effect (MA 2.10.6
helpers/audio.py:861). With the prototype on and no levelling, DJ breaks
come out a few LU quieter than the music, sometimes much more, and news
bulletins quieter still. So, for its own sound effects, the station does what MA
does for a track: it measures each clip and applies one gain toward the same
target. A true-peak limiter stops that gain from clipping.

EXACTLY ONE NORMALISER PER ITEM.
  * A clip levelled here carries a marker: an ID3 TXXX frame named
    MARKER_DESC. ONLY a clip with the marker is inserted as a sound effect
    (url_levelled, ma_queue.program_media_ref).
  * A clip without the marker goes in as an ordinary TRACK, and MA
    normalises it. That happens when the setting was off when the clip was
    produced, when levelling failed, or when the token is retired. Loudness
    is right either way. A failure costs one item that is an Autoplay seed
    again. It never airs a quiet break or a clipped one.
  * With the setting off nothing here runs. Track audio stays byte for byte
    what it was, and MA is its only normaliser.

WHEN. Levelling happens during production, before a token is servable:
render_break levels the bytes it is about to write, and a bulletin is
levelled in its temp slot before it is activated. It never happens later.
Once a URL is servable, MA may be part-way through Range-reading it, and new
bytes under the same URL would corrupt that read. A clip produced while the
setting was off therefore stays unlevelled for its whole life, and airs as a
track.

HOW. ffmpeg is already a hard runtime requirement, so this adds no
dependency.
  1. MEASURE: ebur128 gives the integrated loudness and the true peak
     (4x oversampled). A mono clip is measured as DUAL MONO: MA plays it
     on both channels of a stereo output, and there it is 3 LU louder than
     a plain mono reading says (MEASURE_FILTER).
  2. PLAN: the gain is TARGET_LUFS - I, bounded by MAX_BOOST_DB and
     MAX_CUT_DB. The limiter may take at most MAX_LIMIT_DB off the peaks.
     If more would be needed, the gain is reduced instead: a clip a little
     short of the target is better than a squashed one.
  3. APPLY: volume, then the limiter, but only when the gained peaks would
     pass the ceiling. The limiter runs at 4x the sample rate so that
     inter-sample peaks are caught. The result is re-encoded in the clip's
     own shape (sample rate, channels, CBR bitrate, no Xing header), so
     probing, the duration contract and tail padding behave as before.
  4. VERIFY: the ENCODED result is measured again. It must reach its
     planned loudness, stay under the true-peak ceiling and keep its
     duration. Otherwise LevelError is raised, and the caller keeps the
     unlevelled bytes.

COST. One measuring pass, then one encode-and-measure round per attempt
(usually two: the limiter takes some loudness with the peaks, and a make-up
round puts it back). Memory is the clip itself (a bulletin is a few MB) plus
one ffmpeg process at a time, about 60 MB resident.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("musicbrain")

#: Music Assistant's normalisation target for tracks, as observed in testing.
TARGET_LUFS = -14.0
#: Where the limiter aims the true peak of the levelled clip.
CEILING_DBTP = -1.0
#: A levelled clip whose encoded true peak is above this is refused. This is
#: the headroom margin a lossy decoder needs; at 0 dBTP a clip would clip.
MAX_ACCEPTED_DBTP = -0.5
#: Bounds on the one gain. A boost past MAX_BOOST_DB is raising noise, not
#: speech; any cut is safe, but a clip that needs one this large is broken.
MAX_BOOST_DB = 15.0
MAX_CUT_DB = 20.0
#: The most peak reduction the limiter is allowed. Past this, speech starts
#: to sound squashed, so the gain gives way instead.
MAX_LIMIT_DB = 8.0
#: Below this, the integrated loudness measures noise or silence, not
#: speech: boosting it would be wrong.
MIN_MEASURABLE_LUFS = -50.0
#: EBU R128 gates over 400 ms blocks. A clip shorter than one block has no
#: integrated loudness to correct.
MIN_DURATION_S = 0.4
#: How far the measured result may land from TARGET_LUFS. Further than
#: this, the clip is better off as a track that MA normalises.
LOUDNESS_TOLERANCE_LU = 1.0
#: How much the re-encode may change the clip's length: about two MP3
#: frames of encoder delay and padding. More than this is refused.
MAX_DURATION_DRIFT_S = 0.15
#: The limiter aims this far under CEILING_DBTP: resampling back down and
#: the MP3 encoder each add a little on top of what it lets through.
LIMITER_MARGIN_DB = 0.3
#: A result more than this short of its planned loudness gets make-up gain
#: (the limiter takes some integrated loudness with the peaks).
MAKEUP_ABOVE_LU = 0.5
#: Encode-and-verify rounds per clip: the first, a make-up round and one
#: spare for an encoder overshoot. A long bulletin pays for each.
MAX_ATTEMPTS = 3
#: The limiter runs at this multiple of the clip's rate (true-peak
#: oversampling, as in EBU R128 / ITU-R BS.1770).
OVERSAMPLE = 4
LIMITER_ATTACK_MS = 5
LIMITER_RELEASE_MS = 50
#: One bounded pass, scaled with the clip's length. A wedged binary cannot
#: hold a render or a bulletin fetch open for long.
PASS_TIMEOUT_BASE_S = 30.0
PASS_TIMEOUT_PER_S = 0.5

#: The ID3 TXXX description that marks levelled bytes. Only this module
#: writes it.
MARKER_DESC = "PARLORWAVE_LOUDNESS"
#: 2: mono measured as dual mono (MEASURE_FILTER). A v1 clip was levelled
#: 3 LU too loud if it was mono.
MARKER_VERSION = 2

#: The one loudness reading, used to plan the gain and to verify the
#: result.
#:  - `peak=true`: true peak, 4x oversampled.
#:  - `dualmono=true`: a mono clip measured as it is heard. Music Assistant
#:    plays mono program audio on both channels of a stereo output, and
#:    there it is 3 LU louder than a plain mono reading. Breaks levelled
#:    without it come out about 2-3 dB louder than the music next to
#:    them. ffmpeg applies the
#:    option to mono input only. Stereo bulletins read the same with it,
#:    and are levelled byte for byte as before.
#: Both options exist in the add-on's ffmpeg 5.1; no `framelog` (below).
MEASURE_FILTER = "ebur128=peak=true:dualmono=true"

# Reasons a clip was not levelled. They appear in logs and telemetry.
R_TOO_SHORT = "too_short"
R_UNMEASURABLE = "unmeasurable"
R_NO_TOOL = "ffmpeg_missing"
R_TOOL_FAILED = "ffmpeg_failed"
R_TIMEOUT = "ffmpeg_timeout"
R_SHAPE = "unknown_shape"
R_PEAK = "true_peak_over_ceiling"
R_LOUDNESS = "loudness_off_plan"
R_DURATION = "duration_changed"


class LevelError(RuntimeError):
    """The clip was not levelled. `reason` is one of the R_* codes; the
    caller airs the unlevelled bytes as a track."""

    def __init__(self, reason: str, detail: str = "",
                 report: dict[str, Any] | None = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.report = report or {}


# --- ffmpeg -------------------------------------------------------------------

_SUMMARY_I = re.compile(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+|-?inf)\s*LUFS")
_SUMMARY_TP = re.compile(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-?inf)\s*dBFS")
_TIME = re.compile(r"time=(\d+):(\d+):([\d.]+)")
_AUDIO = re.compile(r"Audio: mp3\w*(?: \([^)]*\))?, (\d+) Hz, ([\w.() ]+?), \w+"
                    r"(?:, (\d+) kb/s)?")


def _timeout(duration_s: float | None) -> float:
    return PASS_TIMEOUT_BASE_S + PASS_TIMEOUT_PER_S * float(duration_s or 600)


#: Below Music Assistant and the rest of the host: on a small machine the
#: encoder must never starve the stream that is playing.
NICE = ["nice", "-n", "10"] if shutil.which("nice") else []


def _run(argv: list[str], audio: bytes,
         duration_s: float | None) -> subprocess.CompletedProcess:
    """One bounded ffmpeg call, the clip on stdin, at low CPU priority.
    argv form only, never shell=True."""
    try:
        proc = subprocess.run([*NICE, "ffmpeg", "-hide_banner", "-nostdin",
                               *argv], input=audio, capture_output=True,
                              timeout=_timeout(duration_s))
    except FileNotFoundError as exc:
        raise LevelError(R_NO_TOOL, str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise LevelError(R_TIMEOUT, f"after {exc.timeout:.0f} s") from exc
    if proc.returncode == 127:            # nice ran, ffmpeg was not found
        raise LevelError(R_NO_TOOL, "ffmpeg not found")
    return proc


def _db(text: str) -> float:
    return -math.inf if "inf" in text else float(text)


def measure(audio: bytes, duration_hint: float | None = None) -> dict[str, Any]:
    """Integrated loudness (LUFS), true peak (dBTP), decoded length and the
    MP3 shape of one clip. Raises LevelError when ffmpeg cannot say."""
    # No `framelog` option. ffmpeg 5.1, the add-on's Debian 12 build, has no
    # `quiet` value for it and rejects the whole filter, so on an earlier build every
    # clip failed to level. Without the option ebur128
    # logs every 100 ms frame: about 0.5 MB of stderr for a 5-minute
    # bulletin, held only for this call. The summary is the last block and
    # is found in the whole text.
    proc = _run(["-i", "pipe:0", "-af", MEASURE_FILTER,
                 "-f", "null", "-"], audio, duration_hint)
    text = proc.stderr.decode("utf-8", "replace")
    m_i, m_tp, times = _SUMMARY_I.search(text), _SUMMARY_TP.search(text), \
        _TIME.findall(text)
    if proc.returncode != 0 or not m_i or not m_tp or not times:
        raise LevelError(R_TOOL_FAILED, text[-200:])
    h, mn, sec = times[-1]
    out: dict[str, Any] = {"i": _db(m_i.group(1)), "tp": _db(m_tp.group(1)),
                           "duration_s": int(h) * 3600 + int(mn) * 60 + float(sec)}
    shape = _AUDIO.search(text)
    if shape:
        layout = shape.group(2).strip()
        out["sample_rate"] = int(shape.group(1))
        out["channels"] = 1 if layout == "mono" else 2
        out["kbps"] = int(shape.group(3)) if shape.group(3) else None
    return out


def plan(i: float, tp: float) -> dict[str, float]:
    """The one gain for a clip measured at `i` LUFS / `tp` dBTP, and how
    much the limiter must take off the peaks after it. Pure arithmetic."""
    gain = max(-MAX_CUT_DB, min(MAX_BOOST_DB, TARGET_LUFS - i))
    limit = max(0.0, tp + gain - CEILING_DBTP)
    if limit > MAX_LIMIT_DB:
        gain -= limit - MAX_LIMIT_DB      # the gain gives way, not the speech
        limit = MAX_LIMIT_DB
    return {"gain_db": round(gain, 2), "limit_db": round(limit, 2),
            "planned_i": round(i + gain, 2)}


def _filters(gain_db: float, limit_db: float, rate: int,
             ceiling_dbfs: float) -> str:
    chain = [f"volume={gain_db:.2f}dB"]
    if limit_db > 0:
        lin = max(0.0625, min(1.0, 10 ** (ceiling_dbfs / 20)))
        chain += [f"aresample={rate * OVERSAMPLE}",
                  f"alimiter=limit={lin:.5f}:attack={LIMITER_ATTACK_MS}"
                  f":release={LIMITER_RELEASE_MS}:level=0:latency=1",
                  f"aresample={rate}"]
    return ",".join(chain)


def _encode(audio: bytes, filters: str, shape: dict[str, Any]) -> bytes:
    """The clip through `filters`, re-encoded in its own shape: raw MPEG
    frames, CBR, no Xing header and no tag (the caller writes the tag)."""
    proc = _run(["-i", "pipe:0", "-af", filters,
                 "-ac", str(shape["channels"]), "-ar", str(shape["sample_rate"]),
                 "-codec:a", "libmp3lame", "-b:a", f"{shape['kbps']}k",
                 "-write_xing", "0", "-id3v2_version", "0",
                 "-map_metadata", "-1", "-f", "mp3", "pipe:1"],
                audio, shape.get("duration_s"))
    if proc.returncode != 0 or not proc.stdout:
        raise LevelError(R_TOOL_FAILED, proc.stderr[-200:].decode(
            "utf-8", "replace"))
    return proc.stdout


def level(audio: bytes) -> tuple[bytes, dict[str, Any]]:
    """`audio` (an MP3, with or without an ID3 tag) levelled toward
    TARGET_LUFS under CEILING_DBTP. Returns the raw MPEG frames, untagged,
    and a report. Raises LevelError and leaves nothing behind when the clip
    cannot be levelled within the contract above."""
    started = time.monotonic()
    src = measure(audio)
    report: dict[str, Any] = {
        "target_lufs": TARGET_LUFS, "in_i": src["i"], "in_tp": src["tp"],
        "in_duration_s": round(src["duration_s"], 3)}
    if src["duration_s"] < MIN_DURATION_S:
        raise LevelError(R_TOO_SHORT, f"{src['duration_s']:.2f} s", report)
    if not math.isfinite(src["i"]) or src["i"] < MIN_MEASURABLE_LUFS:
        raise LevelError(R_UNMEASURABLE, f"I={src['i']}", report)
    if not (src.get("sample_rate") and src.get("channels") and src.get("kbps")):
        raise LevelError(R_SHAPE, "not a CBR MP3 ffmpeg could describe", report)
    tp = src["tp"] if math.isfinite(src["tp"]) else src["i"] + 20.0
    p = plan(src["i"], tp)
    report.update(p)
    gain, limit = p["gain_db"], p["limit_db"]
    ceiling = CEILING_DBTP - LIMITER_MARGIN_DB
    out_bytes, out = b"", {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        out_bytes = _encode(audio, _filters(gain, limit, src["sample_rate"],
                                            ceiling), src)
        out = measure(out_bytes, src["duration_s"])
        report.update({"attempts": attempt, "gain_db": round(gain, 2),
                       "limit_db": round(limit, 2), "out_i": out["i"],
                       "out_tp": out["tp"],
                       "out_duration_s": round(out["duration_s"], 3)})
        if out["tp"] > CEILING_DBTP + 0.1:
            # The encoder overshot the ceiling: limit again, lower by what
            # it added.
            ceiling -= out["tp"] - CEILING_DBTP + 0.2
            limit = max(limit, 0.01)
            continue
        short = p["planned_i"] - out["i"]
        if short <= MAKEUP_ABOVE_LU:
            break
        # The limiter took some integrated loudness with the peaks. Make
        # it up with more gain, within the limiter's cap; stop when the cap
        # leaves nothing to add.
        room = MAX_LIMIT_DB - limit
        add = min(short, room)
        if add < 0.1:
            break
        gain, limit = gain + add, max(limit + add, tp + gain + add - CEILING_DBTP)
    report["ms"] = int((time.monotonic() - started) * 1000)
    if out["tp"] > MAX_ACCEPTED_DBTP:
        raise LevelError(R_PEAK, f"TP={out['tp']}", report)
    if abs(out["i"] - TARGET_LUFS) > LOUDNESS_TOLERANCE_LU:
        # Short of the target even after make-up (a gain or limiter cap
        # held it back): as a track, MA would get it right instead.
        raise LevelError(R_LOUDNESS, f"I={out['i']} target {TARGET_LUFS:g}",
                         report)
    if abs(out["duration_s"] - src["duration_s"]) > MAX_DURATION_DRIFT_S:
        raise LevelError(R_DURATION, f"{src['duration_s']:.3f} -> "
                         f"{out['duration_s']:.3f} s", report)
    return out_bytes, report


def marker(report: dict[str, Any]) -> str:
    """The TXXX value that marks a clip as levelled: what was done, in one
    line a person can read in any tag viewer."""
    return (f"v{MARKER_VERSION} target={report['target_lufs']:g} "
            f"in={report['in_i']:g}/{report['in_tp']:g} "
            f"gain={report['gain_db']:+g} limit={report['limit_db']:g} "
            f"out={report['out_i']:g}/{report['out_tp']:g}")


# --- the insert-time question -------------------------------------------------

def is_levelled(path: Path | str) -> bool:
    """Do the bytes at `path` carry this module's marker?"""
    import audio_probe
    try:
        with Path(path).open("rb") as f:
            head = f.read(audio_probe.MAX_TAG_READ)
    except OSError:
        return False
    return bool(audio_probe.read_user_text(head, MARKER_DESC))


def url_levelled(url: str) -> bool:
    """Will the URL Music Assistant is about to be handed serve LEVELLED
    bytes? Asks the same serving gate the endpoint uses, so a retired token
    (the tombstone), an unknown one, or a file produced while the setting
    was off all answer False, and that item goes in as a track. Any error
    also answers False: the track path is the safe one."""
    import media_ref
    import program_audio
    kind, token = media_ref.programming_ref(url)
    try:
        if kind == "break":
            import dj_engine
            status, path = dj_engine.audio_available(token)
        elif kind == "news":
            import news_bulletins
            status, path = news_bulletins.audio_available(token)
        else:
            return False
        if status != 200 or path is None or program_audio.is_tombstone(path):
            return False
        return is_levelled(path)
    except Exception:  # noqa: BLE001 - unknown is "not levelled"
        logger.warning("program loudness: could not read %s %s", kind,
                       token[:8], exc_info=True)
        return False


def enabled() -> bool:
    """Is program audio produced for the sound-effect path right now?"""
    import settings
    return settings.program_audio_as_sound_effect()


def log_outcome(kind: str, token: str, report: dict[str, Any] | None,
                error: LevelError | Exception | None = None) -> dict[str, Any]:
    """One log line per clip, and the bounded record a producer keeps in
    its telemetry."""
    if error is None and report is not None:
        logger.info(
            "program loudness %s %s levelled: I %.1f -> %.1f LUFS, TP %.1f "
            "-> %.1f dBTP, gain %+.1f dB, limiter %.1f dB, %d ms",
            kind, token[:8], report["in_i"], report["out_i"], report["in_tp"],
            report["out_tp"], report["gain_db"], report["limit_db"],
            report.get("ms", 0))
        return {"levelled": True, **report}
    reason = getattr(error, "reason", None) or R_TOOL_FAILED
    rep = dict(getattr(error, "report", None) or report or {})
    logger.warning(
        "program loudness %s %s NOT levelled (%s: %s); it will be inserted "
        "as a track, which Music Assistant normalises", kind, token[:8],
        reason, str(error)[:160])
    return {"levelled": False, "reason": reason, **rep}
