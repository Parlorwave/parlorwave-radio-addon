"""Home Assistant add-on entry: identity, trust, reachability, then serve.

Runs INSTEAD of `python -m serve` inside the add-on container (see
addon/parlorwave_radio/run.sh). Four bounded steps, none installation-specific:

1. Identity. config.ensure_instance_id persists a random id under the
   add-on's /data. It is what the integration keys its config entry on, so
   it must outlive every restart, rebuild, update and restore -- and must
   never be derived from a hostname, an address or a credential, all of
   which legitimately change.

2. The API key. config.ensure_api_key persists a generated key under the
   same /data, or honours MUSICBRAIN_API_KEY. Never printed.

3. Reachability. The Supervisor's own view of this add-on
   (GET /addons/self/info) gives the hostname Home Assistant and Music
   Assistant resolve, and that is what the service advertises as the base
   of its finite-audio URLs. The container's address is NOT used for that:
   Docker assigns it at each run and it can change, while the hostname is
   re-registered against the current container in the Supervisor's DNS on
   every start. The address remains only as a last-resort fallback for a
   Supervisor that cannot be read, behind the entrypoint's own value.

4. Announcement + serve. The connection -- instance id, hostname, port,
   key -- goes to the Supervisor's discovery API, which only Home Assistant
   Core may read, and Core hands it to the integration's config flow: the
   customer confirms one form and types nothing. Then serve.main().

Discovery is best-effort: a Supervisor that cannot be reached is logged and
the service still starts (the integration can still be added by hand).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import config

SUPERVISOR_URL = "http://supervisor"
DISCOVERY_SERVICE = "parlorwave_radio"
DEFAULT_PORT = 5005
ENV_ADDON_HOSTNAME = "PARLORWAVE_ADDON_HOSTNAME"
ENV_ADDON_IP = "PARLORWAVE_ADDON_IP"

#: Discovery is published only once the service answers HTTP on the
#: loopback. Core validates a discovered flow the moment the Supervisor
#: hands it over (config_flow.async_step_hassio calls GET /instance) and
#: the Supervisor does not re-push an unchanged message, so a message
#: published before uvicorn is bound was an aborted flow and no card until
#: the next Core restart. Bounded: at the deadline it is published anyway
#: -- a late message is a retryable flow, a lost one is not.
READY_DEADLINE_S = 60.0
READY_INTERVAL_S = 0.5

logger = logging.getLogger("musicbrain")


def _supervisor_call(method: str, path: str, token: str, payload: dict | None = None,
                     opener=None, url: str = SUPERVISOR_URL) -> dict | None:
    """One Supervisor API call with this add-on's token. Returns the parsed
    answer, or None when it could not be made (logged, never raised)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{url}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = (exc.read() or b"")[:200].decode("utf-8", "replace")
        logger.warning("Supervisor %s %s refused (HTTP %s): %s",
                       method, path, exc.code, body)
    except Exception as exc:  # noqa: BLE001 - best effort, never fatal
        logger.warning("Supervisor %s %s not answered: %s", method, path, exc)
    return None


def self_info(token: str, opener=None, url: str = SUPERVISOR_URL) -> dict:
    """The Supervisor's own record of THIS add-on (GET /addons/self/info),
    which every add-on may read without `hassio_api`. {} when unavailable."""
    answer = _supervisor_call("GET", "/addons/self/info", token, opener=opener, url=url)
    data = (answer or {}).get("data")
    return data if isinstance(data, dict) else {}


def resolve_hostname(info: dict, env_hostname: str = "") -> str:
    """The name other containers resolve this add-on by.

    The Supervisor is the source of truth and is asked first; the shell
    entrypoint's value (bashio, which reads the same record) is the
    fallback. NEVER constructed from the slug or a repository guess: a
    repository-installed add-on carries a repository prefix that only the
    Supervisor knows."""
    hostname = str((info or {}).get("hostname") or "").strip()
    return hostname or (env_hostname or "").strip()


def advertise_url(hostname: str, port: int, ip_address: str = "") -> str:
    """The base URL Music Assistant fetches finite break audio from.

    The hostname is primary: the Supervisor re-registers it against the
    current container at every start, so it survives the address changing.
    The address is the documented fallback for the case where no hostname
    could be established at all (no Supervisor, no entrypoint value)."""
    host = (hostname or "").strip() or (ip_address or "").strip()
    return f"http://{host}:{int(port)}" if host else ""


def discovery_payload(instance_id: str, hostname: str, port: int, api_key: str,
                      ip_address: str = "") -> dict:
    """What the integration needs to recognise and reach this backend.

    `instance_id` is the identity the config entry is keyed on; host, port
    and api_key are connection DATA, which the integration updates on an
    existing entry when they change."""
    body = {"instance_id": instance_id, "host": hostname,
            "port": int(port), "api_key": api_key}
    if ip_address:
        # diagnostics only: the integration connects by hostname
        body["ip_address"] = ip_address
    return {"service": DISCOVERY_SERVICE, "config": body}


def publish_discovery(token: str, payload: dict, opener=None,
                      url: str = SUPERVISOR_URL) -> dict | None:
    """POST the discovery message. Returns the Supervisor's answer, or None
    when it could not be delivered (logged, never raised)."""
    return _supervisor_call("POST", "/discovery", token, payload=payload,
                            opener=opener, url=url)


def service_answers(port: int, opener=None) -> bool:
    """Does the service accept HTTP on the loopback yet? GET / is the
    unauthenticated liveness route (the add-on watchdog uses it too); any
    HTTP answer means uvicorn is bound and the app has started. A refused
    connection, a timeout or any other failure is 'not yet'."""
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/")
    try:
        with (opener or urllib.request.urlopen)(req, timeout=2) as resp:
            return int(getattr(resp, "status", 200) or 200) < 500
    except Exception:  # noqa: BLE001 - a probe never raises
        return False


def announce_when_serving(token: str, payload: dict, port: int, *,
                          opener=None, probe=None,
                          deadline_s: float = READY_DEADLINE_S,
                          interval_s: float = READY_INTERVAL_S,
                          sleep=time.sleep, clock=time.monotonic) -> str:
    """Publish discovery once the service answers on `port`, or at the
    deadline regardless. Returns "ready" or "deadline". `opener` is the
    Supervisor opener; `probe` replaces the loopback check in tests."""
    check = probe or (lambda: service_answers(port))
    started = clock()
    while not check():
        waited = clock() - started
        if waited >= deadline_s:
            logger.warning("service not answering on port %s after %.0fs; "
                           "publishing discovery anyway", port, deadline_s)
            publish_discovery(token, payload, opener=opener)
            return "deadline"
        sleep(interval_s)
    logger.info("service answering on port %s after %.1fs; publishing "
                "discovery", port, clock() - started)
    publish_discovery(token, payload, opener=opener)
    return "ready"


def start_announcer(token: str, payload: dict, port: int) -> threading.Thread:
    """The deferred announcement on its own daemon thread, so serve.main()
    (which blocks in uvicorn) can start right away."""
    t = threading.Thread(target=announce_when_serving,
                         args=(token, payload, port),
                         name="discovery-announcer", daemon=True)
    t.start()
    return t


def bootstrap(env=None, opener=None, publisher=None) -> dict:
    """Identity, key, reachability, announcement. Returns a summary (no
    secret) for the log and for tests; the caller applies `advertise_url`
    to the environment the service then reads.

    With `publisher`, the discovery message is handed to it instead of
    being posted here (the add-on entry defers it until the service
    answers); the summary then says "deferred"."""
    e = os.environ if env is None else env
    cfg = config.load(e)
    instance_id, id_source = config.ensure_instance_id(cfg)
    key, key_source = config.ensure_api_key(cfg)
    token = (e.get("SUPERVISOR_TOKEN") or "").strip()
    env_hostname = (e.get(ENV_ADDON_HOSTNAME) or "").strip()
    ip_address = (e.get(ENV_ADDON_IP) or "").strip()
    port = int(e.get("MUSICBRAIN_PORT") or DEFAULT_PORT)

    info = self_info(token, opener=opener) if token else {}
    hostname = resolve_hostname(info, env_hostname)
    summary = {"instance_id_source": id_source, "api_key_source": key_source,
               "hostname": hostname or None,
               "hostname_source": ("supervisor" if info.get("hostname")
                                   else ("entrypoint" if env_hostname else None)),
               "ip_address": ip_address or None, "port": port,
               "advertise_url": advertise_url(hostname, port, ip_address) or None,
               "discovery": "skipped"}
    if not token or not hostname:
        logger.warning("not running under the Supervisor (no token/hostname): "
                       "discovery skipped; add the integration by host and key")
        return summary
    payload = discovery_payload(instance_id, hostname, port, key, ip_address)
    if publisher is not None:
        publisher(token, payload)
        summary["discovery"] = "deferred"
        return summary
    answer = publish_discovery(token, payload, opener=opener)
    summary["discovery"] = "published" if answer is not None else "failed"
    return summary


def main() -> int:
    import logging_setup
    logging_setup.configure()
    deferred: dict = {}
    summary = bootstrap(publisher=lambda token, payload: deferred.update(
        token=token, payload=payload))
    if summary.get("advertise_url"):
        # What the service tells Music Assistant to fetch audio from. Set
        # before serve reads the configuration; an explicit environment
        # value (a deliberate override) always wins.
        os.environ.setdefault(config.ENV_ADVERTISE_URL, summary["advertise_url"])
    logger.info("add-on bootstrap: %s", summary)
    if deferred:
        # Announce only once the service answers: serve.main() blocks in
        # uvicorn, so the wait runs beside it and publishes when GET /
        # answers (or at the deadline, never silently not at all).
        start_announcer(deferred["token"], deferred["payload"], summary["port"])
    import serve
    return serve.main()


if __name__ == "__main__":
    sys.exit(main())
