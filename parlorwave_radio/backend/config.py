"""The one typed configuration boundary of the Parlorwave Radio backend.

Everything the service needs from its environment is read HERE, once, into
an immutable Config. No module carries a private default: a value that is
not configured is empty, and the service says so -- at startup for the few
settings it cannot run without, through /health/problems for the rest.

Deployment modes. Running as the Home Assistant add-on (SUPERVISOR_TOKEN
present) the configuration comes from the add-on's own options UI, so a
setting the customer has not entered yet is a REPORTED problem, never a
refusal to start: the add-on must stay up long enough for the integration
to find it and for /health/problems to say what is missing. Everywhere
else the required list below is fatal, exactly as before.

Required (the service refuses to start without them):

    MUSIC_ASSISTANT_URL        the Music Assistant server, e.g.
                               http://music-assistant.local:8095
    MUSIC_ASSISTANT_TOKEN      its API token -- or MUSIC_ASSISTANT_TOKEN_FILE,
                               a file holding the token (read at each use,
                               never logged). As the add-on: reported, not
                               fatal (the customer pastes it after the first
                               start).
    MUSICBRAIN_ADVERTISE_URL   this service's base URL AS MUSIC ASSISTANT
                               REACHES IT, e.g. http://192.0.2.10:5005.
                               Finite break/news audio is fetched from it
                               by Music Assistant, so a loopback address
                               only works when both share a host.
    MUSICBRAIN_BASE_DIR        writable state directory (default /data)

Identity:

    The service keeps a stable instance id at <base_dir>/instance_id --
    generated once, random, and independent of hostname, address, API key
    and host. It is what the Home Assistant integration keys its config
    entry on, so a backend that moves or is renamed stays ONE installation.

Authentication:

    MUSICBRAIN_API_KEY         the X-Api-Key every protected route needs.
                               Unset = the key kept at <base_dir>/api_key,
                               generated on first start (mode 0600).
    MUSICBRAIN_AUTH_DISABLED   "1" opens the API. Development and test
                               harness only; the service warns loudly.

Optional providers -- a blank provider is a normal product state; the
feature is simply unavailable and reported as such:

    OLLAMA_URL                 Ollama /api/generate endpoint
    OLLAMA_MODEL               model for station building / collection tools
    DJ_LLM_MODEL               the DJ writer's model (falls back to OLLAMA_MODEL)
    STATION_PROGRAMMER_MODEL   the optional refill advisor (falls back to the
                               writer's model; settings station.programmer.model
                               overrides)
    SONG_CONTEXT_MODEL         the lyric-theme extractor (same fallback)
    DJ_TTS_URL                 Chatterbox /audio/speech
    DJ_KOKORO_URL + DJ_KOKORO_ENABLED=1
                               the Kokoro sidecar /audio/speech
    Home Assistant (speech through the TTS providers the customer already
    has; render-only, never playback). Two ways in, decided at load:
      add-on mode   SUPERVISOR_TOKEN is present (Home Assistant sets it for
                    an add-on with homeassistant_api: true) -> Core API at
                    http://supervisor/core, that token as the bearer
      external mode HOME_ASSISTANT_URL + HOME_ASSISTANT_TOKEN (or
                    HOME_ASSISTANT_TOKEN_FILE) -- development, or a backend
                    running outside Home Assistant OS. Explicit always wins.
    Neither is required to start; a selected Home Assistant voice with no
    Home Assistant configured is reported by /health/problems.
    LIDARR_URL / LIDARR_API_KEY
    MUSICBRAIN_MUSIC_ROOT      a local music tree for index_library.py
    MUSIC_ASSISTANT_PUBLIC_URL the MA URL browsers reach (artwork); defaults
                               to MUSIC_ASSISTANT_URL

Music Assistant observer (the product's playback authority):

    MA_OBSERVER_MODE           authoritative (default) | shadow | off.
                               Authoritative FEEDS the playback engine from
                               what Music Assistant is playing; shadow only
                               records; off runs nothing. Fresh installs
                               need not set it.
    MA_OBSERVER_QUEUE          auto (default) -- follow the active station,
                               else the selected DJ-enabled output -- or one
                               fixed Music Assistant player id.
    MA_OBSERVER_POLL_S         poll cadence in seconds (clamped by the
                               observer; default 5)

Development only:

    MUSICBRAIN_DEBUG_DOCS      "1" serves the interactive API docs (/docs,
                               /redoc, /openapi.json). Off by default: the
                               product never exposes route enumeration.
    MUSICBRAIN_QUEUE_HYGIENE_CONTROL
                               "1" makes the manual queue-hygiene pass route
                               (POST /queue/hygiene/pass) exist; otherwise it
                               answers 404. A maintenance/testing
                               switch only: it does NOT permit mutation --
                               that is the playback.queue_hygiene setting,
                               which the engine checks on every pass -- and
                               it never touches settings. Off by default.

Compatibility:

    DJ_AUDIO_URL_BASE          pre-dates MUSICBRAIN_ADVERTISE_URL: the full
                               break-audio base (…/broadcast/audio). Honoured
                               when set; MUSICBRAIN_ADVERTISE_URL is the
                               product setting.

Modules that historically read the environment at import time now read
their module constant from `current()` instead; tests keep patching those
per-module constants exactly as before.
"""

from __future__ import annotations

import os
import secrets
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

DEFAULT_BASE_DIR = "/data"
API_KEY_FILE = "api_key"
INSTANCE_ID_FILE = "instance_id"

ENV_MA_URL = "MUSIC_ASSISTANT_URL"
ENV_MA_TOKEN = "MUSIC_ASSISTANT_TOKEN"
ENV_MA_TOKEN_FILE = "MUSIC_ASSISTANT_TOKEN_FILE"
ENV_MA_PUBLIC_URL = "MUSIC_ASSISTANT_PUBLIC_URL"
ENV_ADVERTISE_URL = "MUSICBRAIN_ADVERTISE_URL"
ENV_LEGACY_AUDIO_BASE = "DJ_AUDIO_URL_BASE"
ENV_API_KEY = "MUSICBRAIN_API_KEY"
ENV_AUTH_DISABLED = "MUSICBRAIN_AUTH_DISABLED"
ENV_BASE_DIR = "MUSICBRAIN_BASE_DIR"
ENV_MUSIC_ROOT = "MUSICBRAIN_MUSIC_ROOT"
ENV_OLLAMA_URL = "OLLAMA_URL"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_DJ_LLM_MODEL = "DJ_LLM_MODEL"
ENV_PROGRAMMER_MODEL = "STATION_PROGRAMMER_MODEL"
ENV_SONG_CONTEXT_MODEL = "SONG_CONTEXT_MODEL"
ENV_CHATTERBOX_URL = "DJ_TTS_URL"
ENV_KOKORO_URL = "DJ_KOKORO_URL"
ENV_KOKORO_ENABLED = "DJ_KOKORO_ENABLED"
ENV_LIDARR_URL = "LIDARR_URL"
ENV_HA_URL = "HOME_ASSISTANT_URL"
ENV_HA_TOKEN = "HOME_ASSISTANT_TOKEN"
ENV_HA_TOKEN_FILE = "HOME_ASSISTANT_TOKEN_FILE"
ENV_SUPERVISOR_TOKEN = "SUPERVISOR_TOKEN"
ENV_OBSERVER_MODE = "MA_OBSERVER_MODE"
ENV_OBSERVER_QUEUE = "MA_OBSERVER_QUEUE"
ENV_OBSERVER_POLL_S = "MA_OBSERVER_POLL_S"
ENV_DEBUG_DOCS = "MUSICBRAIN_DEBUG_DOCS"
ENV_QUEUE_HYGIENE_CONTROL = "MUSICBRAIN_QUEUE_HYGIENE_CONTROL"
OBSERVER_OFF, OBSERVER_SHADOW, OBSERVER_AUTHORITATIVE = "off", "shadow", "authoritative"
OBSERVER_MODES = (OBSERVER_OFF, OBSERVER_SHADOW, OBSERVER_AUTHORITATIVE)
OBSERVER_QUEUE_AUTO = "auto"
#: the Alpha product defaults: the backend IS the playback authority and
#: follows the station / the selected DJ-enabled output on its own
DEFAULT_OBSERVER_MODE = OBSERVER_AUTHORITATIVE
DEFAULT_OBSERVER_QUEUE = OBSERVER_QUEUE_AUTO
#: the Supervisor's Core API proxy every add-on reaches (official add-on
#: communication contract: http://supervisor/core/api/ + SUPERVISOR_TOKEN)
ADDON_CORE_URL = "http://supervisor/core"
HA_MODE_ADDON = "addon"
HA_MODE_EXTERNAL = "external"

REQUIRED_ENV = (ENV_MA_URL, f"{ENV_MA_TOKEN} or {ENV_MA_TOKEN_FILE}",
                ENV_ADVERTISE_URL)

SEVERITY_FATAL = "fatal"       # the service will not start
SEVERITY_ERROR = "error"       # running, but the station cannot work
SEVERITY_WARNING = "warning"   # running; one feature is unavailable


class ConfigError(RuntimeError):
    """Startup was refused. `problems` lists why, one entry per setting."""

    def __init__(self, problems: list[dict[str, Any]]) -> None:
        self.problems = problems
        super().__init__("refusing to start: "
                         + "; ".join(p["summary"] for p in problems))


def problem(code: str, severity: str, component: str, summary: str,
            action: str) -> dict[str, Any]:
    """One structured, actionable finding. Never prose to be scraped."""
    return {"code": code, "severity": severity, "component": component,
            "summary": summary, "action": action}


def _clean_url(value: str) -> str:
    return (value or "").strip().rstrip("/")


def url_ok(value: str) -> bool:
    """http(s) with a host. Nothing else is a URL another machine can use."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def is_loopback(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


@dataclass(frozen=True)
class Config:
    base_dir: Path
    music_root: Path | None
    ma_url: str
    ma_token_env: str
    ma_token_file: str
    ma_public_url: str
    advertise_url: str
    legacy_audio_base: str
    api_key: str            # from env or an existing key file; "" until ensured
    api_key_source: str     # "env" | "file" | ""
    instance_id: str        # from an existing file; "" until ensured
    auth_disabled: bool
    ollama_url: str
    ollama_model: str
    dj_writer_model: str
    programmer_model: str
    song_context_model: str
    chatterbox_url: str
    kokoro_url: str
    kokoro_enabled: bool
    lidarr_url: str
    ha_url: str = ""
    ha_mode: str = ""            # "addon" | "external" | ""
    ha_token_env: str = ""
    ha_token_file: str = ""
    addon: bool = False     # running as the Home Assistant add-on
    observer_mode: str = DEFAULT_OBSERVER_MODE     # as configured; may be invalid
    observer_queue: str = DEFAULT_OBSERVER_QUEUE   # "auto" or a fixed player id
    observer_poll_s: str = ""                      # raw; the observer clamps it
    debug_docs: bool = False
    #: the manual queue-hygiene pass route exists. Not permission.
    queue_hygiene_control: bool = False

    # --- derived -------------------------------------------------------
    @property
    def audio_url_base(self) -> str:
        """Where Music Assistant fetches a rendered break: …/broadcast/audio."""
        if self.legacy_audio_base:
            return self.legacy_audio_base
        return f"{self.advertise_url}/broadcast/audio" if self.advertise_url else ""

    @property
    def news_audio_url_base(self) -> str:
        if self.legacy_audio_base:
            root = self.legacy_audio_base.rsplit("/broadcast/audio", 1)[0]
            return f"{root or self.legacy_audio_base}/news/audio"
        return f"{self.advertise_url}/news/audio" if self.advertise_url else ""

    @property
    def api_key_path(self) -> Path:
        return self.base_dir / API_KEY_FILE

    @property
    def instance_id_path(self) -> Path:
        return self.base_dir / INSTANCE_ID_FILE

    def ha_token(self) -> str:
        """The Home Assistant bearer token, never logged."""
        if self.ha_token_env:
            return self.ha_token_env
        if self.ha_token_file:
            try:
                return Path(self.ha_token_file).read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    @property
    def ha_configured(self) -> bool:
        return bool(self.ha_url and self.ha_token())

    def ma_token(self) -> str:
        """The MA token, never logged. Env wins; else the file; else ''."""
        if self.ma_token_env:
            return self.ma_token_env
        if self.ma_token_file:
            try:
                return Path(self.ma_token_file).read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    @property
    def writer(self) -> dict[str, Any]:
        """The DJ writer the ENVIRONMENT configures. NOT the writer in
        force -- ask `optional_services.effective_writer` for that.

        This module is the environment boundary and is deliberately pure,
        so it cannot see a writing model that lives in settings. Reading
        it as the answer to "is a writer configured" is exactly the defect
        the resolver exists to close: an installation that chose its model
        on the Settings screen reported `llm.configured: false` here,
        wrote every break deterministically and never ran the Smart DJ
        planner. Kept for the startup summary (`describe`) and for the
        environment half of the resolution.

        Parlorwave Radio always has a writer: the deterministic one needs
        no configuration, so `configured` is true whether or not an LLM
        is. An LLM (endpoint AND model) is the optional primary."""
        llm = bool(self.ollama_url and self.dj_writer_model)
        return {"configured": True,
                "provider": "ollama" if llm else "deterministic",
                "fallback": "deterministic" if llm else None,
                "llm": {"configured": llm,
                        "provider": "ollama" if llm else None,
                        "model": self.dj_writer_model if llm else None}}

    @property
    def observer_mode_valid(self) -> bool:
        return self.observer_mode in OBSERVER_MODES

    @property
    def observer(self) -> dict[str, Any]:
        """The observer as configured (not as running: see health)."""
        auto = self.observer_queue == OBSERVER_QUEUE_AUTO
        return {"configured_mode": self.observer_mode,
                "valid": self.observer_mode_valid,
                "queue_mode": "auto" if auto else "fixed",
                "queue_id": None if auto else self.observer_queue,
                "default": (self.observer_mode == DEFAULT_OBSERVER_MODE
                            and self.observer_queue == DEFAULT_OBSERVER_QUEUE)}

    @property
    def tts(self) -> dict[str, Any]:
        """The speech providers as configured; the router still decides per
        voice which one speaks."""
        providers = []
        if self.ha_configured:
            providers.append("home_assistant")
        if self.chatterbox_url:
            providers.append("chatterbox")
        if self.kokoro_url and self.kokoro_enabled:
            providers.append("kokoro")
        return {"configured": bool(providers), "providers": providers}


def load(env: Mapping[str, str] | None = None) -> Config:
    """Read the environment into a Config. Pure: nothing is created, nothing
    is generated, nothing is logged. The api_key is whatever the environment
    or an existing key file already holds."""
    e = os.environ if env is None else env
    get = lambda k, d="": (e.get(k) or d).strip()   # noqa: E731 - tiny
    base_dir = Path(get(ENV_BASE_DIR, DEFAULT_BASE_DIR))
    music_root = get(ENV_MUSIC_ROOT)
    api_key = get(ENV_API_KEY)
    source = "env" if api_key else ""
    if not api_key:
        try:
            api_key = (base_dir / API_KEY_FILE).read_text(encoding="utf-8").strip()
            source = "file" if api_key else ""
        except OSError:
            api_key = ""
    try:
        instance_id = (base_dir / INSTANCE_ID_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        instance_id = ""
    ollama_model = get(ENV_OLLAMA_MODEL)
    writer_model = get(ENV_DJ_LLM_MODEL) or ollama_model
    ha_url, ha_mode, ha_token_env, ha_token_file = "", "", "", ""
    if get(ENV_HA_URL):
        ha_url, ha_mode = _clean_url(get(ENV_HA_URL)), HA_MODE_EXTERNAL
        ha_token_env, ha_token_file = get(ENV_HA_TOKEN), get(ENV_HA_TOKEN_FILE)
    elif get(ENV_SUPERVISOR_TOKEN):
        ha_url, ha_mode = ADDON_CORE_URL, HA_MODE_ADDON
        ha_token_env = get(ENV_SUPERVISOR_TOKEN)
    return Config(
        base_dir=base_dir,
        music_root=Path(music_root) if music_root else None,
        ma_url=_clean_url(get(ENV_MA_URL)),
        ma_token_env=get(ENV_MA_TOKEN),
        ma_token_file=get(ENV_MA_TOKEN_FILE),
        ma_public_url=_clean_url(get(ENV_MA_PUBLIC_URL)) or _clean_url(get(ENV_MA_URL)),
        advertise_url=_clean_url(get(ENV_ADVERTISE_URL)),
        legacy_audio_base=_clean_url(get(ENV_LEGACY_AUDIO_BASE)),
        api_key=api_key,
        api_key_source=source,
        instance_id=instance_id,
        auth_disabled=get(ENV_AUTH_DISABLED) == "1",
        ollama_url=get(ENV_OLLAMA_URL),
        ollama_model=ollama_model,
        dj_writer_model=writer_model,
        programmer_model=get(ENV_PROGRAMMER_MODEL) or writer_model,
        song_context_model=get(ENV_SONG_CONTEXT_MODEL) or writer_model,
        chatterbox_url=get(ENV_CHATTERBOX_URL),
        kokoro_url=get(ENV_KOKORO_URL),
        kokoro_enabled=get(ENV_KOKORO_ENABLED) == "1",
        lidarr_url=_clean_url(get(ENV_LIDARR_URL)),
        ha_url=ha_url,
        ha_mode=ha_mode,
        ha_token_env=ha_token_env,
        ha_token_file=ha_token_file,
        addon=bool(get(ENV_SUPERVISOR_TOKEN)),
        observer_mode=(get(ENV_OBSERVER_MODE) or DEFAULT_OBSERVER_MODE).lower(),
        observer_queue=get(ENV_OBSERVER_QUEUE) or DEFAULT_OBSERVER_QUEUE,
        observer_poll_s=get(ENV_OBSERVER_POLL_S),
        debug_docs=get(ENV_DEBUG_DOCS) == "1",
        queue_hygiene_control=get(ENV_QUEUE_HYGIENE_CONTROL) == "1",
    )


_CURRENT: Config | None = None


def current() -> Config:
    """The process configuration, read once from the environment."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load()
    return _CURRENT


def reload(env: Mapping[str, str] | None = None) -> Config:
    """Re-read (tests, and a future add-on options reload)."""
    global _CURRENT
    _CURRENT = load(env)
    return _CURRENT


# --- self-check ------------------------------------------------------------

def required_problems(cfg: Config) -> list[dict[str, Any]]:
    """What stops the service from starting. Empty = it may run."""
    out: list[dict[str, Any]] = []
    try:
        cfg.base_dir.mkdir(parents=True, exist_ok=True)
        writable = os.access(cfg.base_dir, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        out.append(problem(
            "base_dir_unwritable", SEVERITY_FATAL, "storage",
            f"state directory {cfg.base_dir} is not writable",
            f"set {ENV_BASE_DIR} to a writable directory"))
    if not cfg.ma_url:
        out.append(problem(
            "ma_url_missing", SEVERITY_FATAL, "music_assistant",
            "Music Assistant URL is not configured",
            f"set {ENV_MA_URL}, e.g. http://music-assistant.local:8095"))
    elif not url_ok(cfg.ma_url):
        out.append(problem(
            "ma_url_invalid", SEVERITY_FATAL, "music_assistant",
            f"Music Assistant URL is not an http(s) URL: {cfg.ma_url!r}",
            f"set {ENV_MA_URL} to http(s)://host[:port]"))
    if not cfg.ma_token() and not cfg.addon:
        # As the add-on this is NOT fatal: the customer pastes the token into
        # the add-on's options after the first start, and a service that
        # refused to run until then could never be discovered, configured or
        # diagnosed. feature_problems reports it instead, as an error.
        out.append(problem(
            "ma_token_missing", SEVERITY_FATAL, "music_assistant",
            "Music Assistant API token is not configured"
            + (f" ({cfg.ma_token_file} is missing or empty)" if cfg.ma_token_file else ""),
            f"set {ENV_MA_TOKEN}, or {ENV_MA_TOKEN_FILE} to a file holding it"))
    if not cfg.advertise_url:
        out.append(problem(
            "advertise_url_missing", SEVERITY_FATAL, "audio",
            "the URL Music Assistant fetches break audio from is not configured",
            f"set {ENV_ADVERTISE_URL} to this service's base URL as Music "
            "Assistant reaches it, e.g. http://192.0.2.10:5005"))
    elif not url_ok(cfg.advertise_url):
        out.append(problem(
            "advertise_url_invalid", SEVERITY_FATAL, "audio",
            f"advertised URL is not an http(s) URL: {cfg.advertise_url!r}",
            f"set {ENV_ADVERTISE_URL} to http(s)://host[:port]"))
    return out


def feature_problems(cfg: Config) -> list[dict[str, Any]]:
    """What the service can run without, and says so."""
    out: list[dict[str, Any]] = []
    if cfg.addon and not cfg.ma_token():
        # The first-run state of every add-on install: everything else is
        # ready, and this one paste completes it.
        out.append(problem(
            "ma_token_missing", SEVERITY_ERROR, "music_assistant",
            "Music Assistant needs an API token before radio playback can be "
            "observed",
            "create an access token in Music Assistant (Profile > Access "
            "tokens) and paste it into this add-on's Configuration tab"))
    # No LLM is not a problem: the deterministic writer is always there.
    if not cfg.tts["configured"]:
        out.append(problem(
            "tts_unconfigured", SEVERITY_WARNING, "tts",
            "no speech provider is configured: breaks cannot be spoken",
            f"set {ENV_HA_URL} and {ENV_HA_TOKEN} (Home Assistant TTS), "
            f"{ENV_CHATTERBOX_URL}, or {ENV_KOKORO_URL} with {ENV_KOKORO_ENABLED}=1"))
    if not cfg.observer_mode_valid:
        out.append(problem(
            "observer_mode_invalid", SEVERITY_ERROR, "observer",
            f"{ENV_OBSERVER_MODE}={cfg.observer_mode!r} is not one of "
            f"{', '.join(OBSERVER_MODES)}; the observer stays off and no "
            "break can be scheduled",
            f"unset {ENV_OBSERVER_MODE} (default {DEFAULT_OBSERVER_MODE}) or "
            "set a valid mode"))
    elif cfg.observer_mode != OBSERVER_AUTHORITATIVE:
        out.append(problem(
            "observer_disabled", SEVERITY_WARNING, "observer",
            f"the Music Assistant observer is {cfg.observer_mode}: playback is "
            "not followed and no break can be scheduled from it",
            f"unset {ENV_OBSERVER_MODE} (default {DEFAULT_OBSERVER_MODE}) "
            "unless another playback authority feeds this service"))
    if cfg.advertise_url and is_loopback(cfg.advertise_url):
        out.append(problem(
            "advertise_url_loopback", SEVERITY_WARNING, "audio",
            f"advertised URL {cfg.advertise_url} is a loopback address",
            "Music Assistant can only fetch break audio from it when it runs "
            f"on this same host; otherwise set {ENV_ADVERTISE_URL} to a LAN address"))
    return out


def check_required(cfg: Config) -> None:
    fatal = required_problems(cfg)
    if fatal:
        raise ConfigError(fatal)


def ensure_api_key(cfg: Config) -> tuple[str, str]:
    """(key, source). The env key, else the persisted key, else a fresh one
    persisted under base_dir with owner-only permissions. Raises OSError if
    a fresh key cannot be persisted: an unpersisted key would change on
    every restart and lock the integration out."""
    if cfg.api_key:
        return cfg.api_key, cfg.api_key_source
    path = cfg.api_key_path
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key + "\n")
    return key, "generated"


def ensure_instance_id(cfg: Config) -> tuple[str, str]:
    """(instance_id, source). The persisted id, else a fresh random one
    written under base_dir.

    This is the installation's NAME, not a secret and not a fingerprint: a
    random uuid4, carrying nothing about the host, the network, the customer
    or any credential. It is written once and then only read, so an update,
    a rebuild, a reboot and a restore all keep the same identity -- which is
    what lets the Home Assistant integration recognise this backend after
    its hostname, address or API key has changed. Raises OSError if it
    cannot be persisted: an id that changed every start would split one
    installation into many."""
    if cfg.instance_id:
        return cfg.instance_id, "file"
    path = cfg.instance_id_path
    path.parent.mkdir(parents=True, exist_ok=True)
    instance_id = uuid.uuid4().hex
    tmp = path.with_suffix(".tmp")
    tmp.write_text(instance_id + "\n", encoding="utf-8")
    os.replace(tmp, path)
    global _CURRENT
    if _CURRENT is not None and _CURRENT is cfg:
        # the process configuration was read before this file existed; every
        # later reader (health, describe) must see the id, not a blank
        _CURRENT = replace(cfg, instance_id=instance_id)
    return instance_id, "generated"


def describe(cfg: Config) -> dict[str, Any]:
    """A loggable summary. No secrets, ever."""
    return {
        "base_dir": str(cfg.base_dir),
        "instance_id": cfg.instance_id or None,
        "deployment": "addon" if cfg.addon else "standalone",
        "music_assistant_url": cfg.ma_url or None,
        "music_assistant_token": "set" if cfg.ma_token() else "missing",
        "advertise_url": cfg.advertise_url or None,
        "audio_url_base": cfg.audio_url_base or None,
        "auth": "DISABLED" if cfg.auth_disabled else
                (cfg.api_key_source or "generate-on-start"),
        "writer": cfg.writer,
        "tts": cfg.tts,
        "home_assistant": {"mode": cfg.ha_mode or None, "url": cfg.ha_url or None,
                           "token": "set" if cfg.ha_token() else "missing"},
        "observer": cfg.observer,
        "music_root": str(cfg.music_root) if cfg.music_root else None,
        "lidarr": bool(cfg.lidarr_url),
        "debug_docs": cfg.debug_docs,
        "queue_hygiene_control": cfg.queue_hygiene_control,
    }
