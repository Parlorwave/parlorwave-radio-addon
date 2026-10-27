"""Post-render speech audio sanity.

A synthesis API saying 200 is not the same as the audio being sane. A
synthesizer can return success for a
short script while the rendered MP3 holds a few seconds of speech, then
half a minute of internal dead air -- airing as a mostly
silent break. This module is the missing check between "the synthesizer
answered" and "this render may become ready".

CALIBRATION (a sample of known-good breaks across
personalities, voices, and short to long lengths):

    longest INTERNAL silence:  about 1 s    (typically half a second to a second)
    longest trailing silence:  about 1 s    (news-intro pad, deliberate)
    leading silence:           none
    max silence fraction:      about a quarter

    a dead-air render: tens of seconds of internal silence, most of the
    file silent -- fails every check

Thresholds carry large headroom above normal speech so expressive pauses
and deliberate padding always pass:

    MAX_INTERNAL_SILENCE_S = 4.0   (about 4x the normal maximum)
    MAX_SILENCE_FRACTION   = 0.55  (about 2x the normal maximum)
    MIN_AUDIBLE_S          = 1.0   (a "successful" render must contain
                                    at least a breath of actual speech)

Leading/trailing quiet is never a failure by itself (padding is a
feature); it counts only toward the fraction.

ANALYSIS-FAILURE POSTURE: fail OPEN, loudly. If ffmpeg/silencedetect
itself breaks, every render would otherwise fail closed and the DJ would
fall permanently silent because of a tooling problem -- a worse outcome
than the rare pathology this guard exists to catch. The pre-existing
finite-MP3 probe still gates basic validity; an analysis failure logs a
warning and lets the render proceed.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger("musicbrain")

MAX_INTERNAL_SILENCE_S = 4.0
MAX_SILENCE_FRACTION = 0.55
MIN_AUDIBLE_S = 1.0
SILENCE_NOISE_DB = -40
SILENCE_MIN_D = 0.5
EDGE_S = 0.25          # a region touching the ends this closely is edge quiet

_DUR = re.compile(r"Duration: (\d+):(\d+):([\d.]+)")
_TIME = re.compile(r"time=(\d+):(\d+):([\d.]+)")
_START = re.compile(r"silence_start: ([\d.]+)")
_END = re.compile(r"silence_end: ([\d.]+)")


def analyze(mp3: bytes) -> dict[str, Any]:
    """Silence geometry of one finite MP3 (bytes in, stats out). Raises on
    any analysis problem -- the CALLER decides the failure posture."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", "pipe:0",
         "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_D}",
         "-f", "null", "-"],
        input=mp3, capture_output=True, timeout=60)
    text = proc.stderr.decode("utf-8", "replace")
    # Piped input often reports "Duration: N/A"; the decoder's final
    # progress line (time=HH:MM:SS.xx) is the processed duration and is
    # always present on success.
    m = _DUR.search(text)
    times = _TIME.findall(text)
    if proc.returncode != 0 or (not m and not times):
        raise RuntimeError(f"silence analysis failed: {text[-200:]!r}")
    if m:
        total = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                 + float(m.group(3)))
    else:
        h, mn, sec = times[-1]
        total = int(h) * 3600 + int(mn) * 60 + float(sec)
    starts = [float(x) for x in _START.findall(text)]
    ends = [float(x) for x in _END.findall(text)]
    if len(starts) > len(ends):
        ends.append(total)              # silence runs to the end of file
    lead = internal = trail = silence_total = 0.0
    for s, e in zip(starts, ends):
        d = max(0.0, e - s)
        silence_total += d
        if s <= EDGE_S:
            lead = max(lead, d)
        elif e >= total - EDGE_S:
            trail = max(trail, d)
        else:
            internal = max(internal, d)
    return {"duration_s": round(total, 2),
            "max_internal_silence_s": round(internal, 2),
            "leading_silence_s": round(lead, 2),
            "trailing_silence_s": round(trail, 2),
            "silence_fraction": round(silence_total / max(total, 0.1), 3),
            "audible_s": round(max(0.0, total - silence_total), 2)}


def check_speech(mp3: bytes) -> tuple[bool, dict[str, Any]]:
    """(ok, stats). ok=False means the render must not become ready.
    Analysis failure = ok WITH a warning (fail open -- see module doc)."""
    try:
        stats = analyze(mp3)
    except Exception as exc:  # noqa: BLE001 - tooling boundary
        logger.warning("speech audio analysis unavailable (render allowed "
                       "unchecked): %s", str(exc)[:150])
        return True, {"analysis": "unavailable"}
    reasons = []
    if stats["max_internal_silence_s"] > MAX_INTERNAL_SILENCE_S:
        reasons.append("internal_gap")
    if stats["silence_fraction"] > MAX_SILENCE_FRACTION:
        reasons.append("silence_fraction")
    if stats["audible_s"] < MIN_AUDIBLE_S:
        reasons.append("no_audible_speech")
    if reasons:
        stats["reasons"] = reasons
        return False, stats
    return True, stats
