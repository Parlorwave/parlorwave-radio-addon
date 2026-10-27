"""The two optional services, resolved and probed.

WHAT THEY ARE. A voice engine (a Chatterbox-compatible speech service)
speaks the words; an external writing model produces them. They are
separate capabilities that fail separately, and this module keeps them
that way -- one being down is never reported as the other being down.

WHAT THEY ARE NOT. Neither is required. Parlorwave is complete with the
deterministic writer and a Home Assistant text-to-speech voice, and every
path here degrades to exactly that: absent, disabled, unreachable, slow,
or answering nonsense all end in the same place, which is the base product
doing what it always does. Nothing in this module can stop music.

CONFIGURATION BEATS ENVIRONMENT. Both services were environment variables
read once at import, which meant changing either was an add-on restart and
a settings document nobody could see. Stored settings win now; the
environment remains as the fallback so an existing installation keeps
working with nothing re-entered.

SECRETS. A key reaches this module and goes into one Authorization header.
It is never logged, never returned, never put in an error message, and
never included in a diagnostic -- `probe` results carry a reason code and
a short sanitised detail, and the sanitiser is applied to text that came
back from a service we do not control.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("musicbrain")

#: Bounded on purpose. A Test Connection is a person waiting at a screen,
#: not a background job: it gets one attempt, a short deadline, and no
#: retry. A service that needs longer than this to say hello is a service
#: that would have missed its break anyway.
PROBE_TIMEOUT_S = 6.0
#: A candidate that is not there must fail FAST: the finder tries a short
#: list one after another and a person is watching it do so.
FIND_TIMEOUT_S = 1.5
#: Ollama's own default. Named once, here, because two places that each
#: "know" a port is how they come to disagree about it.
OLLAMA_DEFAULT_PORT = 11434
#: Where an Ollama on the same machine as Home Assistant answers. Bounded
#: and fixed: this is a short list of well-known names, NOT a scan. Nothing
#: here enumerates a network, guesses an address range, or tries more than
#: it can name.
OLLAMA_WELL_KNOWN = ("homeassistant", "localhost", "host.docker.internal")
#: Never read an unbounded body from a service we do not control.
MAX_PROBE_BYTES = 64 * 1024
MAX_DETAIL_CHARS = 200

VOICE_ENGINE, WRITER = "voice_engine", "writer"

#: Stable machine reasons. The panel owns the sentences.
OK = "ok"
NOT_CONFIGURED = "not_configured"
DISABLED = "disabled"
UNREACHABLE = "unreachable"
TIMED_OUT = "timed_out"
REFUSED_CREDENTIAL = "refused_credential"
BAD_RESPONSE = "bad_response"
HTTP_ERROR = "http_error"

#: Anything shaped like a credential is removed from a detail string
#: before it travels. A service is free to echo our own Authorization
#: header back inside an error message, and several do.
_SECRETISH = re.compile(
    r"(?i)\b(bearer\s+\S+|sk-[A-Za-z0-9_\-]{8,}|api[_-]?key\s*[=:]\s*\S+"
    r"|authorization\s*[=:]\s*\S+)")


def _settings() -> dict[str, Any]:
    try:
        import settings
        return settings.load().get("services") or {}
    except Exception:  # noqa: BLE001 - settings must never break a break
        logger.warning("optional service settings unreadable", exc_info=True)
        return {}


def resolve(kind: str, cfg: Any = None) -> dict[str, Any]:
    """The effective configuration for one service.

    Stored settings first, the environment second. `enabled` means the service
    is switched on; `usable` means it could actually be tried, which is the
    only question any caller has.

    `cfg` is the process Config to read the environment fallback from.
    Omitted, that is `config.current()`, which is what every runtime
    caller wants; health passes the Config it was handed so a caller that
    STATES an environment is answered about that one.
    """
    stored = (_settings().get(kind) or {}) if kind in (VOICE_ENGINE, WRITER) else {}
    base = str(stored.get("base_url") or "").strip()
    key = str(stored.get("api_key") or "")
    out: dict[str, Any] = {
        "kind": kind,
        "enabled": bool(stored.get("enabled")),
        "base_url": base,
        "api_key": key,
        "source": "settings" if base else "",
    }
    if kind == VOICE_ENGINE:
        out["voice"] = str(stored.get("voice") or "")
        if not base:
            env = _env_voice_engine()
            if env:
                out.update(base_url=env, source="environment")
    else:
        out["provider"] = str(stored.get("provider") or "ollama")
        out["model"] = str(stored.get("model") or "")
        if not base:
            env_url, env_model = _env_writer(cfg)
            if env_url:
                out.update(base_url=env_url, source="environment")
                out["model"] = out["model"] or env_model
    out["usable"] = bool(out["base_url"]) and (
        out["enabled"] or out["source"] == "environment")
    return out


def _env_voice_engine() -> str:
    try:
        import config
        return str(getattr(config.load(), "chatterbox_url", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _env_writer(cfg: Any = None) -> tuple[str, str]:
    """(OLLAMA_URL, DJ_LLM_MODEL-or-OLLAMA_MODEL) from the process
    configuration.

    Two corrections live in these three lines. The model was read from
    `writer_model`, which is not a field of Config -- the attribute is
    `dj_writer_model` -- so the environment fallback has always reported
    an EMPTY model, and `getattr`'s default hid it. And the process
    boundary is `current()`, not `load()`: `load()` re-reads os.environ
    and therefore walks straight past `config.reload(env)`, which is how
    every test states the environment it means.
    """
    try:
        if cfg is None:
            import config
            cfg = config.current()
        return (str(getattr(cfg, "ollama_url", "") or ""),
                str(getattr(cfg, "dj_writer_model", "") or ""))
    except Exception:  # noqa: BLE001
        return "", ""


# --- the writing model the RUNTIME will actually use -------------------------

#: Ollama's generation route. The Settings screen asks for an ADDRESS (a
#: base, which is also what Test Connection derives /api/tags from); the
#: OLLAMA_URL environment variable has always been the full endpoint. This
#: is the one place those two shapes are reconciled.
OLLAMA_GENERATE_PATH = "/api/generate"

#: What the writer IS, as opposed to what is written down.
WRITER_MODE_LLM = "llm"
WRITER_MODE_DETERMINISTIC = "deterministic"

#: Where the usable configuration came from. `fallback` is the honest name
#: for "nowhere": the deterministic writer is not configured, it simply is.
SOURCE_SETTINGS = "settings"
SOURCE_ENVIRONMENT = "environment"
SOURCE_FALLBACK = "fallback"

#: Why a writing model that LOOKS configured is not one the runtime can use.
#: These are the states that used to be indistinguishable from "configured".
WRITER_NO_ENDPOINT = "no_endpoint"
WRITER_NO_MODEL = "no_model"
WRITER_DISABLED = "switched_off"
WRITER_PROVIDER_UNSUPPORTED = "provider_unsupported"

#: The only provider the DJ runtime can speak. `openai_compatible` is a
#: valid thing to SAVE and a valid thing to Test, and the runtime still
#: cannot write a break with it -- so it resolves to the deterministic
#: writer and says why, rather than being counted as a working model.
RUNTIME_PROVIDER = "ollama"


def generate_url(base: str) -> str:
    """The /api/generate route of an Ollama-shaped address.

    The Settings screen asks for an address and stores what was typed, so
    a saved writer is usually a bare `http://box:11434`. Handing that
    straight to `providers.ollama_generate` POSTs to Ollama's ROOT, which
    answers 405 -- measured against a real Ollama server.
    Every break would have fallen back to deterministic copy with a
    transport error, which is indistinguishable from the model being down
    and is not what "I selected this model" is supposed to mean.

    Idempotent, and it completes rather than rewrites: an address that
    already names a route keeps it, exactly as `_probe_writer` derives
    /api/tags from the same field.
    """
    s = str(base or "").strip()
    if not s:
        return ""
    return s.split("/api/")[0].rstrip("/") + OLLAMA_GENERATE_PATH


def effective_writer(cfg: Any = None) -> dict[str, Any]:
    """THE writing model configuration, for every caller that has one.

    One resolver, so the writer, the Smart DJ planner and the health
    surface cannot answer differently -- which is the whole defect this
    exists to close. Settings could name a model, Test Connection could
    reach it, and the runtime still wrote every break deterministically
    while health reported no writer configured, because "is a writer
    configured" was read from the environment and nothing else.

    PRECEDENCE, and why:

      1. STORED SETTINGS, when the owner switched the service on and gave
         it both an address and a model. A person who chose a model on the
         screen has said which model they mean, and an environment
         variable set once at install time does not get to overrule that.
      2. THE ENVIRONMENT (OLLAMA_URL + DJ_LLM_MODEL/OLLAMA_MODEL), for
         every installation that predates the Settings screen. Nobody has
         to migrate.
      3. NEITHER -- the deterministic writer, which is the product and not
         a degraded mode.

    A SAVED-BUT-SWITCHED-OFF service does NOT fall through to the
    environment. `resolve` stops looking once settings name an address,
    and that is deliberate: the off switch is the most explicit statement
    the owner can make, and quietly honouring a legacy variable instead
    would be exactly the settings/runtime split in the other direction.

    BOTH HALVES OR NEITHER. An endpoint without a model is not a usable
    writer, and neither half is ever borrowed across sources -- a settings
    address is never paired with an environment model. Half a
    configuration resolves to the deterministic writer with a reason,
    never to a writer that is going to fail at the next break.

    Never raises: a writer that cannot be resolved is the deterministic
    writer, which is always available.
    """
    try:
        res = resolve(WRITER, cfg)
    except Exception:  # noqa: BLE001 - settings must never cost a break
        logger.warning("writing model unresolvable", exc_info=True)
        res = {}
    source = res.get("source") or ""
    provider = str(res.get("provider") or RUNTIME_PROVIDER)
    model = str(res.get("model") or "").strip()
    base = str(res.get("base_url") or "").strip()
    key = str(res.get("api_key") or "")
    if source == SOURCE_ENVIRONMENT:
        # EVERY HALF FROM ONE SOURCE. `resolve` lets a settings model name
        # and a settings key stand in when the ADDRESS came from the
        # environment, which is reasonable for the Settings screen's own
        # inventory and wrong for the runtime. A model typed beside an
        # address that was never saved gets asked of a server that was
        # never told about it, and answers 404 -- and the key is worse
        # than wrong: it is a credential for one service, sent in an
        # Authorization header to a different host. The environment
        # configures an address and a model, and carries no key.
        model = str(_env_writer(cfg)[1] or "").strip()
        key = ""
        # OLLAMA_URL names an Ollama. A `provider` left in the settings
        # document from a service whose address was never saved does not
        # describe it, and health would print that name beside an
        # environment endpoint it has nothing to do with.
        provider = RUNTIME_PROVIDER
    # The environment variable IS the generate endpoint and has been for
    # as long as it has existed; only a settings ADDRESS needs completing.
    endpoint = base if source == SOURCE_ENVIRONMENT else generate_url(base)

    out: dict[str, Any] = {
        "mode": WRITER_MODE_DETERMINISTIC,
        "source": SOURCE_FALLBACK,
        "provider": provider,
        "endpoint": "",
        "base_url": base,
        "model": "",
        "api_key": "",
        "usable": False,
        "reason": "",
    }
    if not base:
        out["reason"] = NOT_CONFIGURED
        return out
    if not res.get("usable"):
        # An address is saved but the service is switched off. Configured,
        # deliberately not in use, and said so rather than shown as broken.
        out["reason"] = WRITER_DISABLED
        return out
    if source == SOURCE_SETTINGS and provider != RUNTIME_PROVIDER:
        out["reason"] = WRITER_PROVIDER_UNSUPPORTED
        return out
    if not endpoint:
        out["reason"] = WRITER_NO_ENDPOINT
        return out
    if not model:
        out["reason"] = WRITER_NO_MODEL
        return out
    out.update(mode=WRITER_MODE_LLM, source=source or SOURCE_FALLBACK,
               endpoint=endpoint, model=model,
               api_key=key, usable=True)
    return out


def writer_runtime(cfg: Any = None) -> dict[str, Any]:
    """`effective_writer` with the key removed, for anything that is going
    to be rendered, logged or returned over HTTP.

    The address stays -- it is already on the Settings screen the owner
    typed it into -- and the key never leaves this module.
    """
    out = dict(effective_writer(cfg))
    out["key_set"] = bool(out.pop("api_key", ""))
    return out


def sanitize(text: Any) -> str:
    """A detail string fit to leave the building.

    Two jobs. It removes anything credential-shaped, because a service we
    do not control may echo our own Authorization header inside its error
    text. And it strips the angle brackets, because this string is
    rendered on a screen and a remote service's error message is untrusted
    input like any other -- the panel escapes it too, and that is on
    purpose: neither layer is the only one.
    """
    s = str(text or "").strip()
    s = _SECRETISH.sub("[redacted]", s)
    s = s.replace("<", "").replace(">", "")
    s = re.sub(r"\s+", " ", s)
    return s[:MAX_DETAIL_CHARS]


def _request(url: str, *, api_key: str = "", payload: dict | None = None,
             timeout: float = PROBE_TIMEOUT_S) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(MAX_PROBE_BYTES)


def _failure(exc: Exception, timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    """One exception, as a reason a person can act on. The exception text
    is sanitised before it is allowed anywhere near a response."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read(MAX_PROBE_BYTES).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        reason = (REFUSED_CREDENTIAL if exc.code in (401, 403) else HTTP_ERROR)
        return {"ok": False, "reason": reason, "status": exc.code,
                "detail": sanitize(body or exc.reason)}
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return {"ok": False, "reason": TIMED_OUT,
                "detail": f"No answer within {timeout:g} seconds."}
    return {"ok": False, "reason": UNREACHABLE, "detail": sanitize(exc)}


def normalize_url(text: Any, *, default_port: int | None = None,
                  default_path: str = "") -> str:
    """What a person typed, as an address that can be tried.

    They are reading a screen that says "Address", and they type what they
    know: the machine's name, its IP, a name and a port, or a full URL.
    Refusing the short forms teaches nothing -- the address IS reachable
    and the product simply would not look at it. So a missing scheme becomes http, a
    missing port becomes the service's own default, and a missing path
    becomes the one this kind of service serves. Anything already given is
    left exactly as it was: this completes an address, it never rewrites
    one.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    try:
        parts = urllib.parse.urlsplit(s)
    except Exception:  # noqa: BLE001 - a typo is not a crash
        return s
    netloc, path = parts.netloc, parts.path
    # The default port completes a BARE address -- a name or an IP -- and
    # nothing else. An address that already carries a path is one somebody
    # copied from a service that told them what to use, and adding a port
    # to that is rewriting it rather than completing it.
    bare = path.strip("/") == ""
    if default_port and bare and netloc and not parts.port and not netloc.endswith(":"):
        netloc = f"{netloc}:{default_port}"
    if default_path and path.strip("/") == "":
        path = default_path
    return urllib.parse.urlunsplit((parts.scheme or "http", netloc, path,
                                    parts.query, parts.fragment))


def probe(kind: str, override: dict[str, Any] | None = None, *,
          timeout: float | None = None) -> dict[str, Any]:
    """Can this service be reached and used, right now?

    ONE attempt, one short deadline, no retry, and nothing cached: a Test
    Connection is a person waiting at a screen. It reads; it never renders
    speech, never writes a break, never touches a queue or a player, and
    cannot change what the station is about to do.

    WITH AN OVERRIDE IT TESTS WHAT IS ON THE SCREEN. The button sits under
    the address field, so it must answer for the address in that field --
    not for the one that was saved the last time somebody pressed Save. A
    test that silently reports on a different address than the one being
    read is worse than no test, and that is what this did: type a new
    Ollama address, press Test, and be told about the old one. The
    override is used for this call and stored nowhere; saving is still
    Save's job, and a typed address is deliberately treated as enabled
    because asking to test it is asking to test it.
    """
    cfg = resolve(kind)
    if override:
        typed = normalize_url(
            override.get("base_url"),
            default_port=(OLLAMA_DEFAULT_PORT
                          if kind == WRITER and cfg.get("provider") != "openai_compatible"
                          else None))
        if typed:
            cfg = {**cfg, "base_url": typed, "source": "typed", "enabled": True}
        key = override.get("api_key")
        if isinstance(key, str) and key:
            cfg = {**cfg, "api_key": key}
        for field in ("provider", "model", "voice"):
            if isinstance(override.get(field), str) and override[field]:
                cfg = {**cfg, field: override[field]}
    if not cfg["base_url"]:
        return {"ok": False, "reason": NOT_CONFIGURED, "kind": kind,
                "detail": "No address has been set."}
    if not cfg["enabled"] and cfg["source"] != "environment":
        return {"ok": False, "reason": DISABLED, "kind": kind,
                "detail": "This service is switched off."}
    deadline = PROBE_TIMEOUT_S if timeout is None else timeout
    try:
        if kind == VOICE_ENGINE:
            result = _probe_voice_engine(cfg, deadline)
        else:
            result = _probe_writer(cfg, deadline)
    except Exception as exc:  # noqa: BLE001 - any network shape
        result = _failure(exc, deadline)
    result["kind"] = kind
    result.setdefault("source", cfg["source"])
    return result


def voices_url(base: str) -> str:
    """A Chatterbox voice inventory lives beside its speech endpoint. The
    ONE derivation: the probe here and every voice-library call
    (providers.chatterbox_voices_url) use it, so they cannot disagree."""
    base = base.rstrip("/")
    for tail in ("/audio/speech", "/speech", "/v1/audio/speech"):
        if base.endswith(tail):
            return base[: -len(tail)] + "/voices"
    return base + "/voices"


_voices_url = voices_url


def _probe_voice_engine(cfg: dict[str, Any],
                        timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    status, body = _request(_voices_url(cfg["base_url"]),
                            api_key=cfg["api_key"],
                            timeout=timeout)
    voices = _voice_names(body)
    if voices is None:
        return {"ok": False, "reason": BAD_RESPONSE, "status": status,
                "detail": "The address answered, but not with a voice list."}
    return {"ok": True, "reason": OK, "status": status, "voices": voices,
            "detail": f"{len(voices)} voice{'' if len(voices) == 1 else 's'} "
                      f"available."}


def _voice_names(body: bytes) -> list[str] | None:
    """Voice names out of whatever shape the service answered with.

    Tolerant of the three shapes these services actually use, and strict
    about what it returns: names only, bounded in count and length, and
    every one a plain string. A service is untrusted input.
    """
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("voices", "data", "results"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    if rows is None:
        return None
    names = []
    for row in rows[:200]:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = row.get("name") or row.get("id") or row.get("voice") or ""
        else:
            continue
        name = sanitize(name)[:120]
        if name:
            names.append(name)
    return names


def _probe_writer(cfg: dict[str, Any],
                  timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    base = cfg["base_url"].rstrip("/")
    if cfg.get("provider") == "openai_compatible":
        url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    else:
        url = base.split("/api/")[0] + "/api/tags"
    # The deadline is stated at the CALL SITE, not left to a default: a
    # bound nobody can see in the code that relies on it is a bound that
    # quietly changes.
    status, body = _request(url, api_key=cfg["api_key"], timeout=timeout)
    models = _model_names(body)
    if models is None:
        return {"ok": False, "reason": BAD_RESPONSE, "status": status,
                "detail": "The address answered, but not with a model list."}
    wanted = (cfg.get("model") or "").strip()
    known = bool(wanted) and any(wanted == m or m.startswith(wanted + ":")
                                 for m in models)
    detail = f"{len(models)} model{'' if len(models) == 1 else 's'} available."
    if wanted and not known:
        # NOT a failure: a service may serve a model it does not list, and
        # refusing here would be this module overruling the owner.
        detail += f" '{sanitize(wanted)}' is not among them."
    return {"ok": True, "reason": OK, "status": status, "models": models,
            "model_listed": known, "detail": detail}


def _model_names(body: bytes) -> list[str] | None:
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    rows = None
    if isinstance(data, dict):
        for key in ("models", "data"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    elif isinstance(data, list):
        rows = data
    if rows is None:
        return None
    names = []
    for row in rows[:200]:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = row.get("name") or row.get("id") or row.get("model") or ""
        else:
            continue
        name = sanitize(name)[:120]
        if name:
            names.append(name)
    return names


def find_writer(typed: Any = "", *, provider: str = "ollama",
                probe_fn=None) -> dict[str, Any]:
    """Look for an Ollama-shaped service at a SHORT LIST OF NAMED PLACES.

    This is not a scanner and must never become one. It tries, in order:
    the address on the screen completed with the default port, then three
    well-known names for "on the same machine as Home Assistant". Every
    candidate is named in the answer, whether it replied or not, so the
    screen can say what was looked at rather than "not found".

    A network scan is the obvious next idea and is deliberately refused:
    it is slow, it looks like a port sweep to anything watching the
    network, and a house with a /16 would never finish. If the service is
    somewhere else, the person types where -- which is why the typed
    address is tried first and why the field accepts a bare IP.
    """
    probe_fn = probe_fn or probe
    candidates: list[str] = []
    typed_url = normalize_url(typed, default_port=OLLAMA_DEFAULT_PORT)
    if typed_url:
        candidates.append(typed_url)
    for host in OLLAMA_WELL_KNOWN:
        url = f"http://{host}:{OLLAMA_DEFAULT_PORT}"
        if url not in candidates:
            candidates.append(url)
    tried: list[dict[str, Any]] = []
    for url in candidates:
        result = probe_fn(WRITER, {"base_url": url, "provider": provider},
                          timeout=FIND_TIMEOUT_S)
        row = {"base_url": url, "ok": bool(result.get("ok")),
               "reason": result.get("reason")}
        if result.get("ok"):
            row["models"] = result.get("models") or []
            tried.append(row)
            return {"ok": True, "found": url, "models": row["models"],
                    "tried": tried, "searched": candidates}
        tried.append(row)
    return {"ok": False, "found": None, "models": [], "tried": tried,
            "searched": candidates}


def status() -> dict[str, Any]:
    """What each service IS, without contacting either.

    For the readiness document and the Settings screen, which are read
    often and must never make a network call to render. No key, no
    address-with-credentials, nothing that is not already safe to show.
    """
    out = {}
    for kind in (VOICE_ENGINE, WRITER):
        cfg = resolve(kind)
        out[kind] = {
            "enabled": cfg["enabled"],
            "configured": bool(cfg["base_url"]),
            "usable": cfg["usable"],
            "source": cfg["source"],
            "key_set": bool(cfg["api_key"]),
            "model": cfg.get("model", ""),
            "voice": cfg.get("voice", ""),
            "provider": cfg.get("provider", ""),
        }
    return out
