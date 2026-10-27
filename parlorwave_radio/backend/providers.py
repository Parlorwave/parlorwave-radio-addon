"""The ONE place that knows how to talk to the current text/speech
providers (Ollama for generation, Chatterbox for speech).

Business logic (dj_engine, llm_station_builder, missing_albums) decides
WHAT to ask -- model, prompt, format, voice -- and how to interpret the
decoded reply; this module owns the HTTP mechanics and the single
authoritative default per endpoint setting. render_break keeps its
llm_fn/tts_fn injection seam and merely defaults to thin wrappers over
these functions, so tests and future providers plug in above this layer
without touching the engine.

Deliberately NOT a plugin framework: the two current providers, plain
functions. urllib.request.urlopen must stay late-bound (module-attribute
call, never `from urllib.request import urlopen`) -- several tests patch
it globally.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import writer_gate

# One authoritative copy of each shared default. The env variable names
# are long-standing operational contract -- never renamed here. OLLAMA_MODEL
# is the station/collector default; the DJ writer's own model knob is the
# dj_writer role in inference.llm_adapter, which falls back to it.
# Read through the one configuration boundary; an unconfigured provider is
# "" here and a reported, non-fatal absence everywhere else.
import config as _config

_cfg = _config.current()
OLLAMA_URL = _cfg.ollama_url
OLLAMA_MODEL = _cfg.ollama_model
DJ_TTS_URL = _cfg.chatterbox_url                 # Chatterbox /audio/speech
DJ_TTS_TIMEOUT_S = float(os.environ.get("DJ_TTS_TIMEOUT_S", "90"))
# Voice inventory. Derived from DJ_TTS_URL (…/audio/speech -> …/voices) unless
# an installation points elsewhere.
DJ_TTS_VOICES_URL = os.environ.get("DJ_TTS_VOICES_URL", "")
# Kokoro: the curated built-in DJ voices, served by a dedicated sidecar that
# owns the native runtime. Unset URL or DJ_KOKORO_ENABLED != "1"
# means the installation behaves EXACTLY as it did before Kokoro existed --
# that default is the rollout kill switch, and it is off until an owner
# turns it on.
HA_URL = _cfg.ha_url                              # Home Assistant Core API base
DJ_KOKORO_URL = _cfg.kokoro_url                  # Kokoro /audio/speech
DJ_KOKORO_TIMEOUT_S = float(os.environ.get("DJ_KOKORO_TIMEOUT_S", "90"))
DJ_KOKORO_ENABLED = "1" if _cfg.kokoro_enabled else "0"


def ollama_probe(*, timeout: float = 3.0, url: str | None = None) -> dict[str, Any]:
    """READ-ONLY reachability check of an Ollama server for diagnostics:
    one GET of its tag list (…/api/tags, derived from the endpoint).
    Never raises.

    `url` is the caller's RESOLVED endpoint when it has one, so health
    probes the server the writer will actually call. Omitted, this is
    exactly the OLLAMA_URL check it has always been."""
    endpoint = url if url is not None else OLLAMA_URL
    out: dict[str, Any] = {"ok": False, "url": endpoint, "error": ""}
    if not endpoint:
        out["error"] = "not configured"
        return out
    base = endpoint.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            out["ok"] = 200 <= getattr(resp, "status", 200) < 300
    except Exception as exc:  # noqa: BLE001 - one boundary, one meaning
        out["error"] = str(exc)[:200]
    return out


def _require_llm(model: str, url: str | None = None) -> None:
    """An unconfigured LLM is a plain, named failure -- never a malformed
    request to an empty URL.

    `url` is the caller's RESOLVED endpoint when it has one, so a writer
    configured in settings is not refused for an environment variable it
    was introduced to replace."""
    if not (url or OLLAMA_URL):
        raise RuntimeError("no writing model is configured")
    if not model:
        raise RuntimeError("no writing model name is configured")


def ollama_generate(prompt: str, model: str, *, timeout: float,
                    format: str | dict | None = None,
                    url: str | None = None,
                    api_key: str | None = None,
                    options: dict[str, Any] | None = None) -> dict[str, Any]:
    """One non-streaming Ollama /api/generate call; returns the DECODED
    BODY. Callers keep their own response-field semantics (the DJ writer
    tolerates a missing 'response'; the station builder requires it), so
    this returns the dict rather than flattening those differences.
    Raises on network/HTTP/JSON failure exactly like the inlined clients
    it replaced -- each caller owns its failure posture. `format` may be
    the string "json" (legacy JSON mode) or a full JSON schema dict
    (Ollama >= 0.5 structured outputs: the decoder is constrained to the
    schema, which is what makes the LLM programmer's output shape
    reliable). `options` is Ollama's per-request runtime map (num_ctx,
    num_predict, ...); omitted, the payload is exactly what it always was
    -- the Smart DJ planner is the one caller that states its context
    window, and the free-text writer deliberately does not."""
    _require_llm(model, url)
    payload: dict[str, Any] = {"model": model, "prompt": prompt,
                               "stream": False}
    if format:
        payload["format"] = format
    if options:
        payload["options"] = dict(options)
    # `url` and `api_key` are the CONFIGURED writer, resolved by the caller
    # at call time. Omitted, this is exactly the environment-configured
    # call it has always been.
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url or OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers=headers)
    # LIVE demand: background story preparation yields to -- and cancels
    # for -- every call that comes through here (writer_gate). Signalled,
    # never queued: this never waits on anything background work holds.
    with writer_gate.live():
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)


class GenerationCancelled(Exception):
    """A background generate given up for live demand (or its own wall-time
    cap). Nothing it streamed may be used."""


def ollama_generate_background(prompt: str, model: str, *,
                               max_s: float, format: str | dict | None = None,
                               url: str | None = None,
                               api_key: str | None = None,
                               options: dict[str, Any] | None = None,
                               cancel: Callable[[], bool] | None = None,
                               poll_s: float = 0.05) -> dict[str, Any]:
    """One STREAMING generate for background work only (story preparation).

    It is NOT live demand: it never enters writer_gate.live(). A watcher
    checks `cancel()` every `poll_s` and the wall time against `max_s`; on
    either, it shuts the socket down -- which interrupts a read blocked in
    prompt evaluation as well as one between tokens -- and the call raises
    GenerationCancelled. Only a COMPLETE reply (`done: true`) is returned,
    as {"response": <all tokens>, ...final chunk fields}; a partial reply is
    discarded, never returned. Transport and HTTP failures raise like
    ollama_generate."""
    import http.client
    import socket
    import threading
    import time
    _require_llm(model, url)
    target = urllib.parse.urlsplit(url or OLLAMA_URL)
    payload: dict[str, Any] = {"model": model, "prompt": prompt,
                               "stream": True}
    if format:
        payload["format"] = format
    if options:
        payload["options"] = dict(options)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn_cls = (http.client.HTTPSConnection if target.scheme == "https"
                else http.client.HTTPConnection)
    conn = conn_cls(target.hostname, target.port, timeout=max_s)
    path = target.path or "/"
    if target.query:
        path += "?" + target.query
    stop = threading.Event()
    why: list[str] = []
    t0 = time.monotonic()

    def watch():
        while not stop.wait(poll_s):
            reason = ("cancelled" if cancel is not None and cancel() else
                      "time_cap" if time.monotonic() - t0 >= max_s else None)
            if reason:
                why.append(reason)
                try:
                    if conn.sock is not None:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    watcher = threading.Thread(target=watch, name="bg-generate-watch",
                               daemon=True)
    watcher.start()
    parts: list[str] = []
    try:
        conn.request("POST", path, body=json.dumps(payload).encode("utf-8"),
                     headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            raise urllib.error.HTTPError(url or OLLAMA_URL, resp.status,
                                         resp.reason, resp.headers, None)
        final: dict[str, Any] | None = None
        for line in resp:
            if why:
                break
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            parts.append(str(chunk.get("response") or ""))
            if chunk.get("done"):
                final = chunk
                break
        if why or final is None:
            raise GenerationCancelled(why[0] if why else "incomplete")
        return {**final, "response": "".join(parts)}
    except GenerationCancelled:
        raise
    except Exception as exc:
        if why:
            raise GenerationCancelled(why[0]) from exc
        raise
    finally:
        stop.set()
        conn.close()


def ollama_warm(model: str, *, timeout: float,
                keep_alive: str = "10m",
                options: dict[str, Any] | None = None) -> None:
    """Empty-prompt keep_alive poke to load the model off the critical
    path. The response body is deliberately never read (the original warm
    path never parsed it). Raises on failure -- the caller owns the
    best-effort semantics. `options` as for ollama_generate: a warm must
    load the model with the context window the calls after it will ask
    for, or the first of them reloads it."""
    _require_llm(model)
    payload: dict[str, Any] = {"model": model, "prompt": "", "stream": False,
                               "keep_alive": keep_alive}
    if options:
        payload["options"] = dict(options)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with writer_gate.live():               # a warm is live demand too
        with urllib.request.urlopen(req, timeout=timeout):
            pass


def chatterbox_synthesize(text: str, voice: str = "", *,
                          timeout: float | None = None) -> bytes:
    """Speech audio from Chatterbox, using the LIVE-VERIFIED contract
    (chatterbox-tts-api):

        POST {DJ_TTS_URL}   body {"input": <text>, "voice": <voice name>}
        -> 200 audio/wav (16-bit PCM mono 24kHz)

    Chatterbox emits WAV only (format hints are ignored); render_break
    detects the RIFF magic and transcodes to the finite MP3 the break
    contract requires. An empty voice omits the key entirely (the service
    default voice speaks -- no installation's asset name is ever assumed).
    Unset URL = hard failure, which the break lifecycle converts into a
    safe 'failed' (music untouched)."""
    url, key = chatterbox_endpoint()
    if not url:
        raise RuntimeError("no voice engine is configured")
    payload: dict[str, Any] = {"input": text}
    if voice:
        payload["voice"] = voice
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout or DJ_TTS_TIMEOUT_S) as resp:
        return resp.read()


def chatterbox_endpoint() -> tuple[str, str]:
    """(url, key) for the voice engine, resolved AT CALL TIME.

    DJ_TTS_URL is read once at import, which meant the only way to point
    the station at a different speech service was an add-on restart -- and
    the address lived in an environment variable nobody could see from the
    product. Stored settings win now; the module constant remains as the
    fallback so an installation that has always used the variable keeps
    working with nothing re-entered.

    Never raises, and never logs the key.
    """
    try:
        import optional_services
        cfg = optional_services.resolve(optional_services.VOICE_ENGINE)
        if cfg["usable"]:
            return cfg["base_url"], cfg["api_key"]
    except Exception:  # noqa: BLE001 - settings must not break speech
        pass
    # The environment as it is NOW: the module constant was captured at
    # import, so a configuration reload never reached it.
    return DJ_TTS_URL or _config.current().chatterbox_url, ""


def chatterbox_voices_url() -> str:
    """The voice library beside the configured speech endpoint, derived by
    the SAME rule the Settings probe uses (optional_services.voices_url),
    so an address that tests fine there is the address enrollment uses."""
    if DJ_TTS_VOICES_URL:
        return DJ_TTS_VOICES_URL
    url, _key = chatterbox_endpoint()
    if not url:
        return ""
    import optional_services
    return optional_services.voices_url(url)


def _engine_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The voice engine's key, on every call that reaches it -- the library
    reads and writes as well as speech. Never logged."""
    headers = dict(extra or {})
    _url, key = chatterbox_endpoint()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def chatterbox_voices(*, timeout: float = 5.0) -> list[dict[str, Any]]:
    """The voices the TTS service can speak with, LIVE-VERIFIED contract
    (chatterbox-tts-api):

        GET {base}/voices -> {"voices": [{"name": ..., "filename": ...,
                                          "language": ..., "aliases": [...]}]}

    Returns the raw voice records (name is the value POST /audio/speech
    accepts as "voice"). Raises when no URL is configured or the service
    cannot answer; the caller decides what an unknown inventory means."""
    url = chatterbox_voices_url()
    if not url:
        raise RuntimeError("DJ_TTS_URL not configured")
    req = urllib.request.Request(url, headers=_engine_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    voices = doc.get("voices") if isinstance(doc, dict) else None
    if not isinstance(voices, list):
        raise RuntimeError("voice inventory malformed")
    return [v for v in voices if isinstance(v, dict) and v.get("name")]

def ha_configured() -> bool:
    """Home Assistant TTS is a usable route: a base URL and a token."""
    return bool(_config.current().ha_configured)


def ha_tts_synthesize(text: str, entity_id: str, *,
                      timeout: float | None = None) -> bytes:
    """Speech from a Home Assistant TTS entity, render-only (ha_tts).
    Raises ha_tts.HAError with a named category; never plays anything."""
    import ha_tts
    return ha_tts.synthesize(text, entity_id, timeout=timeout)


def kokoro_enabled() -> bool:
    """The rollout kill switch. Both halves must be true: an operator has
    turned Kokoro on AND an endpoint exists. Anything else is the existing
    Chatterbox installation, unchanged."""
    return DJ_KOKORO_ENABLED == "1" and bool(DJ_KOKORO_URL)


def kokoro_synthesize(text: str, voice: str, *,
                      speed: float | None = None,
                      silence_scale: float | None = None,
                      timeout: float | None = None) -> bytes:
    """Speech audio from the Kokoro sidecar, using the sidecar's provider
    contract:

        POST {DJ_KOKORO_URL}  body {"input": <text>, "voice": <kokoro id>}
        -> 200 audio/wav, finite valid 24 kHz WAV

    Same wire shape as Chatterbox on purpose, so render_break's existing
    RIFF-detect-and-transcode path is reached unchanged and the finite-MP3
    publication contract is untouched.

    The voice is REQUIRED and must be an engine voice name: the contract
    forbids a silent fallback, because a service that quietly picks a
    speaker puts the wrong person on air. Unset URL is a hard failure,
    which the break lifecycle converts into a safe 'failed' (music
    untouched)."""
    if not DJ_KOKORO_URL:
        raise RuntimeError("DJ_KOKORO_URL not configured")
    if not voice:
        raise RuntimeError("kokoro requires an explicit voice")
    payload: dict[str, Any] = {"input": text, "voice": voice}
    # delivery knobs travel only when set, so an untuned voice is the
    # exact baseline request
    if speed is not None:
        payload["speed"] = float(speed)
    if silence_scale is not None:
        payload["silence_scale"] = float(silence_scale)
    req = urllib.request.Request(
        DJ_KOKORO_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req,
                                timeout=timeout or DJ_KOKORO_TIMEOUT_S) as resp:
        return resp.read()


def kokoro_voices_url() -> str:
    if DJ_KOKORO_URL.endswith("/audio/speech"):
        return DJ_KOKORO_URL[: -len("/audio/speech")] + "/voices"
    return ""


def kokoro_voices(*, timeout: float = 5.0) -> list[dict[str, Any]]:
    """What the sidecar says it can speak:

        GET {base}/voices -> {"voices": [{"name", "speaker_id",
                                          "language", "label"}]}

    Doubles as the readiness probe -- a sidecar that cannot list its
    voices has not loaded its model, so it cannot synthesize either."""
    url = kokoro_voices_url()
    if not url:
        raise RuntimeError("DJ_KOKORO_URL not configured")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    voices = doc.get("voices") if isinstance(doc, dict) else None
    if not isinstance(voices, list):
        raise RuntimeError("kokoro voice inventory malformed")
    return [v for v in voices if isinstance(v, dict) and v.get("name")]


def chatterbox_upload_voice(name: str, data: bytes, filename: str,
                            *, timeout: float = 30.0) -> dict[str, Any]:
    """Upload one voice sample: POST {base}/voices multipart
    (voice_name, voice_file, language) -- the service's own documented
    contract (OpenAPI, verified live). Returns the decoded body; raises
    on any failure. The caller owns naming and validation."""
    base = chatterbox_voices_url()
    if not base:
        raise RuntimeError("DJ_TTS_URL not configured")
    boundary = "----jarvisvoice" + os.urandom(8).hex()
    body = b""
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="voice_name"\r\n\r\n{name}\r\n').encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="language"\r\n\r\nen\r\n').encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="voice_file"; filename="{filename}"\r\n'
             "Content-Type: application/octet-stream\r\n\r\n").encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base, data=body, method="POST",
        headers=_engine_headers(
            {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chatterbox_delete_voice(name: str, *, timeout: float = 15.0) -> None:
    """DELETE {base}/voices/{name}. Raises on failure; a 404 (already
    absent) is treated as success -- deletion is idempotent."""
    base = chatterbox_voices_url()
    if not base:
        raise RuntimeError("DJ_TTS_URL not configured")
    req = urllib.request.Request(f"{base}/{urllib.parse.quote(name)}",
                                 method="DELETE", headers=_engine_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        raise


def chatterbox_download_voice(name: str, *, timeout: float = 30.0,
                              limit: int = 64 * 1024 * 1024) -> bytes:
    """GET {base}/voices/{name}/download -- the stored recording's own
    bytes (Chatterbox TTS API 2.1.0, listed in its OpenAPI).
    Read-only: the Voice Library keeps a copy of a slot's recording with
    it before anything replaces or removes that recording. Raises on any
    failure, and on a body over `limit` rather than truncating it."""
    base = chatterbox_voices_url()
    if not base:
        raise RuntimeError("DJ_TTS_URL not configured")
    req = urllib.request.Request(f"{base}/{urllib.parse.quote(name)}/download",
                                 headers=_engine_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("voice download larger than the limit")
    return data


def chatterbox_rename_voice(name: str, new_name: str, *,
                            timeout: float = 15.0) -> None:
    """PUT {base}/voices/{name} (form: new_name) -- the service's own
    rename (Chatterbox TTS API 2.1.0, read from its code: the
    file and its library entry move together; 404 for an unknown name,
    409 when new_name is held). Raises on any failure, 404 included: a
    replace that swaps recordings must know the rename happened."""
    base = chatterbox_voices_url()
    if not base:
        raise RuntimeError("DJ_TTS_URL not configured")
    body = urllib.parse.urlencode({"new_name": new_name}).encode()
    req = urllib.request.Request(
        f"{base}/{urllib.parse.quote(name)}", data=body, method="PUT",
        headers=_engine_headers(
            {"Content-Type": "application/x-www-form-urlencoded"}))
    with urllib.request.urlopen(req, timeout=timeout):
        return
