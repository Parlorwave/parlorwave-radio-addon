"""Home Assistant text-to-speech as a Parlorwave speech provider.

RENDER ONLY. Parlorwave asks Home Assistant to synthesize speech and takes
the AUDIO BYTES back; it never asks Home Assistant to play anything. The
finite program element stays Parlorwave's (normalized, tokenized, served
from /broadcast/audio and inserted queue-native) exactly like every other
provider's speech. `tts.speak` and any media_player are never involved.

The surface, verified against the Home Assistant 2026.8 source
(homeassistant/components/tts/__init__.py):

    POST /api/tts_get_url            TextToSpeechUrlView, requires_auth
        {"engine_id": "<tts.* entity id>", "message": "<text>",
         "cache": false, ["language": ...], ["options": {...}]}
        -> 200 {"url": "<HA base>/api/tts_proxy/<token>.<ext>",
                "path": "/api/tts_proxy/<token>.<ext>"}
        -> 400 {"message"|"error": ...}  unknown engine, bad language/option
        `engine_id` accepts a tts.* entity id (get_engine_instance);
        cache=false keeps the result in memory only (no file written);
        the extension is options.preferred_format or "mp3" (_DEFAULT_FORMAT);
        the render itself happens when the proxy URL is first read.
    GET  /api/tts_proxy/<token>      TextToSpeechView, public (token-scoped),
        streams the audio with the content type of its extension; 404 for
        an unknown token; a provider failure surfaces as an HTTP error.

Authentication is the one configured in config.py: the add-on's
SUPERVISOR_TOKEN against http://supervisor/core (official add-on Core API
proxy) or an explicit HOME_ASSISTANT_URL + HOME_ASSISTANT_TOKEN.

Safety: the only locations ever fetched are the `path` Home Assistant
returned, resolved against the CONFIGURED base -- never a URL a caller
supplied, never another origin, never a redirect (refused outright).
Responses are size-bounded, time-bounded, and must actually be audio.
Nothing here logs the token, the URL, or a provider's error body.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import config as _config

logger = logging.getLogger("musicbrain")

RENDER_TIMEOUT_S = 45.0          # a Pi-class Piper needs a few seconds
PROBE_TIMEOUT_S = 5.0
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MIN_AUDIO_BYTES = 128
LIST_TTL_S = 60.0
PROXY_PREFIX = "/api/tts_proxy/"
ENTITY_RE = re.compile(r"^tts\.[a-z0-9_]+$")
_PATH_RE = re.compile(r"^/api/tts_proxy/[A-Za-z0-9_\-]+\.[a-z0-9]{2,5}$")

KIND_UNCONFIGURED = "unconfigured"
KIND_UNREACHABLE = "unreachable"
KIND_AUTH = "auth_rejected"
KIND_NOT_FOUND = "not_found"
KIND_BAD_REQUEST = "bad_request"
KIND_RENDER_FAILED = "render_failed"
KIND_NOT_AUDIO = "not_audio"
KIND_TOO_LARGE = "too_large"
KIND_TIMEOUT = "timeout"
KIND_MALFORMED = "malformed_response"
KIND_REDIRECT = "redirect_refused"


class HAError(RuntimeError):
    """One named failure. `kind` is one of the KIND_* categories; the
    message is safe to log and to show (no token, no URL, no body)."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(f"home assistant {kind}: {message}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is refused, never followed: the only origin we talk to
    is the configured one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def _open(req: urllib.request.Request, timeout: float):
    """The one transport seam (tests replace it)."""
    return _OPENER.open(req, timeout=timeout)


def audio_kind(data: bytes) -> str:
    """'wav' | 'mp3' | 'ogg' | 'flac' | '' by magic bytes, never by name."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF
                              and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    return ""


def _read_bounded(resp, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = resp.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HAError(KIND_TOO_LARGE,
                          f"audio exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_detail(exc: urllib.error.HTTPError) -> str:
    """Home Assistant's own short error text, when the body is JSON with a
    message/error field; never the raw body, never anything else."""
    try:
        body = exc.read(2048)
        data = json.loads(body.decode("utf-8", "replace"))
        msg = data.get("message") or data.get("error")
        return f": {str(msg)[:160]}" if msg else ""
    except Exception:  # noqa: BLE001 - detail is a nicety
        return ""


def _request(method: str, path: str, *, body: dict[str, Any] | None = None,
             timeout: float) -> tuple[int, Any, bytes]:
    """One authenticated call to the configured Home Assistant, at a path
    WE chose. Every failure is an HAError with a category; nothing about
    the token or the URL survives into the message."""
    cfg = _config.current()
    if not cfg.ha_configured:
        raise HAError(KIND_UNCONFIGURED, "Home Assistant is not configured")
    headers = {"Authorization": f"Bearer {cfg.ha_token()}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(cfg.ha_url + path, data=data,
                                 headers=headers, method=method)
    try:
        with _open(req, timeout) as resp:
            return (int(getattr(resp, "status", 200)), resp.headers,
                    _read_bounded(resp, MAX_AUDIO_BYTES))
    except HAError:
        raise
    except urllib.error.HTTPError as exc:
        code = int(exc.code)
        if code in (301, 302, 303, 307, 308):
            raise HAError(KIND_REDIRECT,
                          "Home Assistant answered with a redirect; refused") from None
        if code in (401, 403):
            raise HAError(KIND_AUTH,
                          f"Home Assistant rejected the token (HTTP {code})") from None
        if code == 404:
            raise HAError(KIND_NOT_FOUND, f"HTTP 404 for {path.split('?')[0]}") from None
        detail = _safe_detail(exc)
        raise HAError(KIND_BAD_REQUEST if code == 400 else KIND_RENDER_FAILED,
                      f"HTTP {code}{detail}") from None
    except (TimeoutError, socket.timeout):
        raise HAError(KIND_TIMEOUT, f"no answer within {timeout:.0f}s") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise HAError(KIND_TIMEOUT, f"no answer within {timeout:.0f}s") from None
        raise HAError(KIND_UNREACHABLE,
                      f"cannot reach Home Assistant ({str(exc.reason)[:120]})") from None
    except OSError as exc:
        raise HAError(KIND_UNREACHABLE,
                      f"cannot reach Home Assistant ({type(exc).__name__})") from None


def _same_origin(url: str, base: str) -> bool:
    a, b = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base)
    return (a.scheme, a.hostname, a.port or {"http": 80, "https": 443}.get(a.scheme)) == \
           (b.scheme, b.hostname, b.port or {"http": 80, "https": 443}.get(b.scheme))


def _audio_path(data: Any, base: str) -> str:
    """The proxy path from the render answer, trusted only in the shape
    Home Assistant produces: a relative /api/tts_proxy/<token>.<ext>. An
    absolute `url` is used only when it is on the configured origin, and
    even then only its path is kept."""
    if not isinstance(data, dict):
        raise HAError(KIND_MALFORMED, "render answer is not an object")
    path = data.get("path")
    if not path and data.get("url"):
        url = str(data["url"])
        if not _same_origin(url, base):
            raise HAError(KIND_MALFORMED,
                          "audio location is not on the configured Home Assistant origin")
        path = urllib.parse.urlsplit(url).path
    if not isinstance(path, str) or not _PATH_RE.match(path):
        raise HAError(KIND_MALFORMED, "render answer carries no usable audio path")
    return path


def synthesize(text: str, entity_id: str, *, language: str | None = None,
               options: dict[str, Any] | None = None,
               timeout: float | None = None) -> bytes:
    """Speech bytes for `text` from the Home Assistant TTS entity, in
    whatever format that provider produces (the caller normalizes)."""
    entity_id = (entity_id or "").strip()
    if not ENTITY_RE.match(entity_id):
        raise HAError(KIND_BAD_REQUEST, f"not a tts entity id: {entity_id!r}")
    if not (text or "").strip():
        raise HAError(KIND_BAD_REQUEST, "nothing to say")
    budget = float(timeout or RENDER_TIMEOUT_S)
    body: dict[str, Any] = {"engine_id": entity_id, "message": text,
                            "cache": False}
    if language:
        body["language"] = language
    if options:
        body["options"] = dict(options)
    _status, _hdrs, raw = _request("POST", "/api/tts_get_url", body=body,
                                   timeout=budget)
    try:
        answer = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise HAError(KIND_MALFORMED, "render answer is not JSON") from None
    path = _audio_path(answer, _config.current().ha_url)
    _status, hdrs, audio = _request("GET", path, timeout=budget)
    ctype = str(hdrs.get("Content-Type") or "").split(";")[0].strip().lower()
    kind = audio_kind(audio)
    if not kind and (ctype.startswith("text/") or ctype.startswith("application/json")
                     or ctype.startswith("application/xml") or not ctype.startswith("audio/")):
        raise HAError(KIND_NOT_AUDIO,
                      f"Home Assistant returned {ctype or 'no content type'}, not audio")
    if len(audio) < MIN_AUDIO_BYTES:
        raise HAError(KIND_RENDER_FAILED, "Home Assistant returned no usable audio")
    return audio


_LIST_CACHE: dict[str, Any] = {"at": 0.0, "value": None}


def reset_cache() -> None:
    _LIST_CACHE.update(at=0.0, value=None)


def _entity_record(state: dict[str, Any]) -> dict[str, Any]:
    attrs = state.get("attributes") or {}
    eid = str(state.get("entity_id") or "")
    return {"entity_id": eid,
            "name": str(attrs.get("friendly_name") or eid.split(".", 1)[-1]),
            "state": str(state.get("state") or ""),
            "available": str(state.get("state") or "") != "unavailable"}


def list_entities(*, timeout: float | None = None) -> list[dict[str, Any]]:
    """Every tts.* entity Home Assistant has, from the ordinary states API
    (cached briefly: the shelf is read often). Raises HAError."""
    now = time.monotonic()
    if _LIST_CACHE["value"] is not None and now - _LIST_CACHE["at"] < LIST_TTL_S:
        return list(_LIST_CACHE["value"])
    _status, _hdrs, raw = _request("GET", "/api/states",
                                   timeout=timeout or PROBE_TIMEOUT_S)
    try:
        states = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise HAError(KIND_MALFORMED, "states answer is not JSON") from None
    out = sorted((_entity_record(s) for s in states
                  if isinstance(s, dict) and str(s.get("entity_id", "")).startswith("tts.")),
                 key=lambda r: r["entity_id"])
    _LIST_CACHE.update(at=now, value=out)
    return list(out)


def entity_state(entity_id: str, *, timeout: float | None = None) -> dict[str, Any]:
    """One tts.* entity's state. KIND_NOT_FOUND when it does not exist."""
    if not ENTITY_RE.match(entity_id or ""):
        raise HAError(KIND_BAD_REQUEST, f"not a tts entity id: {entity_id!r}")
    _status, _hdrs, raw = _request("GET", f"/api/states/{entity_id}",
                                   timeout=timeout or PROBE_TIMEOUT_S)
    try:
        return _entity_record(json.loads(raw.decode("utf-8")))
    except Exception:  # noqa: BLE001
        raise HAError(KIND_MALFORMED, "state answer is not JSON") from None


def probe(*, timeout: float | None = None) -> dict[str, Any]:
    """READ-ONLY reachability and credential check (GET /api/). Never
    raises; never logs the token."""
    cfg = _config.current()
    out: dict[str, Any] = {"ok": False, "configured": bool(cfg.ha_configured),
                           "mode": cfg.ha_mode or None, "url": cfg.ha_url or None,
                           "kind": "", "error": ""}
    if not cfg.ha_configured:
        out["kind"] = KIND_UNCONFIGURED
        out["error"] = "not configured"
        return out
    try:
        _request("GET", "/api/", timeout=timeout or PROBE_TIMEOUT_S)
        out["ok"] = True
    except HAError as exc:
        out["kind"], out["error"] = exc.kind, str(exc)
    return out
