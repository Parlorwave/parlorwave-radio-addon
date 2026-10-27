"""The product-facing health surface: structured, actionable problems.

Configuration facts come from config; the one live check is a read-only
Music Assistant probe. Nothing here plans, renders or touches transport,
and an unreachable MA is a REPORTED problem, never an exception.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import autoplay_watch
import config as _config

AUTH_STATUSES = (401, 403)


def ma_problem(probe: dict[str, Any]) -> dict[str, Any] | None:
    """The MA probe's outcome as a problem, or None when MA answered."""
    if probe.get("ok"):
        return None
    status = probe.get("http_status")
    if status in AUTH_STATUSES:
        return _config.problem(
            "ma_auth_rejected", _config.SEVERITY_ERROR, "music_assistant",
            f"Music Assistant rejected the API token (HTTP {status})",
            f"check {_config.ENV_MA_TOKEN} / {_config.ENV_MA_TOKEN_FILE} "
            "against a token issued by this Music Assistant")
    if not probe.get("url"):
        return _config.problem(
            "ma_url_missing", _config.SEVERITY_FATAL, "music_assistant",
            "Music Assistant URL is not configured",
            f"set {_config.ENV_MA_URL}")
    return _config.problem(
        "ma_unreachable", _config.SEVERITY_ERROR, "music_assistant",
        f"Music Assistant at {probe.get('url')} did not answer: "
        f"{probe.get('error') or 'no response'}",
        "check the URL, that Music Assistant is running, and that this "
        "service can reach it on the network")


def effective_writer(cfg: _config.Config | None = None) -> dict[str, Any]:
    """The writing model the RUNTIME will actually use.

    Asks the resolver the WRITER asks, so this surface and the break
    cannot answer differently. Never raises, never touches the key.

    `cfg` is threaded into the environment half of the resolution so a
    caller that was HANDED a configuration is answered about that one;
    with none, this is `config.current()`, which is the writer's own
    view and therefore what production reports."""
    try:
        import optional_services
        return optional_services.writer_runtime(cfg)
    except Exception:  # noqa: BLE001 - a settings failure is not a health crash
        return {"mode": "deterministic", "source": "fallback", "usable": False,
                "endpoint": "", "base_url": "", "model": "", "provider": "",
                "reason": "", "key_set": False}


def writer_block(writer: dict[str, Any],
                 llm_probe: dict[str, Any] | None) -> dict[str, Any]:
    """`features.writer`: what writes a break, where it came from, and
    whether it answered.

    `cfg.writer` was the ENVIRONMENT alone, exactly as `cfg.tts` once
    was -- so an installation that configured its model the way
    the product tells it to reported `llm.configured: false` while
    Settings showed a saved, tested model. This is the same correction
    applied to the other optional service.

    The legacy nested `llm` block and `llm_reachable` are kept verbatim:
    the integration's readiness card reads them, and it should start
    telling the truth without being edited. `mode`, `source`, `model` and
    `reachable` are the flat, honest names beside them.

    Deliberately NOT here: the address and the key. `source` says where
    the configuration came from, which is what a diagnostic needs.
    """
    llm = writer["mode"] == "llm"
    reachable = bool(llm_probe.get("ok")) if llm_probe else None
    return {
        "configured": True,       # the deterministic writer always is
        "provider": writer["provider"] if llm else "deterministic",
        "fallback": "deterministic" if llm else None,
        "mode": writer["mode"],
        "source": writer["source"],
        "model": writer["model"] or None,
        "reachable": reachable,
        "reason": writer["reason"] or None,
        "llm": {"configured": llm,
                "provider": writer["provider"] if llm else None,
                "model": writer["model"] or None},
        "llm_reachable": reachable,
    }


def writer_problem(writer: dict[str, Any], llm_probe: dict[str, Any] | None
                   ) -> dict[str, Any] | None:
    """An LLM that is configured but not answering is a WARNING: the
    deterministic writer keeps the station operational.

    The remedy names the place the address actually came from. Telling
    somebody who chose a model on the Settings screen to go and check an
    environment variable they have never set sends them looking for a
    configuration that does not exist."""
    if writer["mode"] != "llm" or not llm_probe or llm_probe.get("ok"):
        return None
    where = ("the writing model in Settings"
             if writer["source"] == "settings" else _config.ENV_OLLAMA_URL)
    return _config.problem(
        "optional_writer_unavailable", _config.SEVERITY_WARNING, "writer",
        f"the LLM writer at {llm_probe.get('url')} is not answering "
        f"({llm_probe.get('error') or 'no response'}); breaks are written "
        "deterministically until it returns",
        f"check {where} and that the server is running")


def writer_settings_problem(writer: dict[str, Any]) -> dict[str, Any] | None:
    """Settings that NAME a writing model the runtime will not use.

    The failure this closes was silent: Settings could
    model a service the DJ never called, and nothing said so. A saved
    address that resolves to the deterministic writer is now stated --
    once, as a warning, because the station is entirely operational.

    A switched-off service is NOT this. That is a choice, and it is
    already shown as one on the screen that made it."""
    reason = writer.get("reason")
    if writer["mode"] == "llm" or not writer.get("base_url"):
        return None
    import optional_services
    if reason == optional_services.WRITER_PROVIDER_UNSUPPORTED:
        return _config.problem(
            "writer_provider_unsupported", _config.SEVERITY_WARNING, "writer",
            f"the writing model is saved as '{writer['provider']}', which the "
            "DJ cannot write breaks with; breaks are written deterministically",
            "choose the ollama provider for the writing model, or leave it "
            "switched off")
    if reason == optional_services.WRITER_NO_MODEL:
        return _config.problem(
            "writer_model_missing", _config.SEVERITY_WARNING, "writer",
            "the writing model has an address but no model name, so no model "
            "is used; breaks are written deterministically",
            "choose a model for the writing model in Settings")
    return None


def _voice_refs() -> dict[str, Any]:
    """dj.persona.voice.refs, or {} when settings cannot be read."""
    try:
        import tts_router
        return tts_router._refs()
    except Exception:  # noqa: BLE001 - a settings failure is not a health crash
        return {}


def voice_selected(refs: dict[str, Any] | None) -> bool:
    """PURE. Has anybody ever chosen a voice on this installation?

    The router has no 'nothing chosen' answer: with empty refs it falls
    through to the Chatterbox path this installation has always had, which
    is the right routing decision and the wrong DIAGNOSIS. A fresh install
    has not chosen badly -- it has not chosen at all, and that is a setup
    state, not a fault. This is the one place the two are told apart."""
    import tts_router
    return any(str((refs or {}).get(k) or "").strip()
               for k in (tts_router.HA_REF, tts_router.KOKORO_REF,
                         tts_router.CHATTERBOX_REF))


def _voice_engine_configured() -> bool:
    """Would `providers.chatterbox_synthesize` have somewhere to send text?

    It asks the resolver SPEECH asks, in the same settings-then-environment
    order, so health and speech cannot answer differently. Never raises,
    never touches the key."""
    try:
        import providers as _providers
        url, _key = _providers.chatterbox_endpoint()
        return bool(url)
    except Exception:  # noqa: BLE001 - a settings failure is not a health crash
        return False


def effective_tts(cfg: _config.Config) -> dict[str, Any]:
    """The speech providers that could ACTUALLY speak right now.

    `cfg.tts` is read from the ENVIRONMENT alone -- "chatterbox" is
    listed only when DJ_TTS_URL is set -- but the voice engine stopped
    being an environment variable when it moved onto the Settings screen,
    and `providers.chatterbox_endpoint` has resolved stored settings first
    ever since. So an installation that configured its engine the way the
    product tells it to could speak every break through Chatterbox while this
    surface reported that Chatterbox was not configured. The readiness
    model turned that into a blocker -- *the chosen voice speaks through
    chatterbox, which is not configured; breaks cannot be spoken* -- and
    Settings showed **Cannot speak** on a station that was, audibly,
    speaking.

    One resolver answers both questions now. Nothing here contacts the
    service: this is "is it configured", not "is it reachable", which is
    what Test Connection is for."""
    block = dict(cfg.tts)
    names = list(block.get("providers") or [])
    if "chatterbox" not in names and _voice_engine_configured():
        names.append("chatterbox")
    block["providers"] = names
    block["configured"] = bool(names)
    return block


def selected_provider_problems(cfg: _config.Config, route: dict[str, Any] | None,
                               refs: dict[str, Any] | None = None,
                               tts: dict[str, Any] | None = None
                               ) -> list[dict[str, Any]]:
    """What is wrong with the speech SELECTION, as opposed to the engines.

    Two different states, deliberately two different problems:

      NOTHING CHOSEN YET -- the fresh install. A warning, worded as setup,
      naming no provider: the customer has done nothing wrong and the
      internal name of the engine the router would have fallen back to is
      not a fact about their installation.

      CHOSEN, BUT ITS ENGINE IS GONE -- something that WAS configured has
      stopped working. An error, and the provider is named because whoever
      reads the service log needs it. Nothing customer-facing renders this
      summary: the integration's readiness model rewrites it."""
    providers_configured = (effective_tts(cfg) if tts is None else tts)["providers"]
    if not providers_configured:
        return []        # tts_unconfigured already covers having no engine
    if not voice_selected(_voice_refs() if refs is None else refs):
        # Nobody has chosen, and a voice that can speak is picked for them
        # (voice_library.automatic_dj_voice): a fresh installation has no
        # voice-selection step left to do.
        import voice_library
        if voice_library.automatic_dj_voice():
            return []
        return [_config.problem(
            "tts_voice_unselected", _config.SEVERITY_WARNING, "tts",
            "no DJ voice has been chosen yet; breaks cannot be spoken until "
            "one is",
            "choose a voice in the ParlorWave Radio panel, under Booth")]
    provider = (route or {}).get("provider")
    if not provider or provider in providers_configured:
        return []
    return [_config.problem(
        "tts_selected_provider_unconfigured", _config.SEVERITY_ERROR, "tts",
        f"the selected voice speaks through {provider}, which is not configured; "
        "breaks cannot be spoken",
        "pick a voice from a configured provider (a Home Assistant voice under "
        "Voice) or configure that provider")]


def outputs_problems(dj: dict[str, Any]) -> list[dict[str, Any]]:
    """The DJ-enabled output allow-list (settings.dj.outputs.enabled) is
    what lets a break be scheduled at all. Empty is a valid configuration
    state, and it is DIAGNOSED: an error when the DJ is switched on (the
    customer expects breaks and none can come), a warning otherwise
    (setup is incomplete). A selected default output outside the
    allow-list is a warning: the observer will not follow it."""
    out: list[dict[str, Any]] = []
    outputs = (dj or {}).get("outputs") or {}
    enabled = [o for o in (outputs.get("enabled") or []) if o]
    selected = (outputs.get("default_request_output") or "").strip()
    dj_on = bool((dj or {}).get("enabled"))
    if not enabled:
        out.append(_config.problem(
            "no_outputs_enabled",
            _config.SEVERITY_ERROR if dj_on else _config.SEVERITY_WARNING,
            "outputs",
            "No DJ-enabled Music Assistant outputs have been selected"
            + ("; the DJ is on but no break can be scheduled" if dj_on else ""),
            "select the outputs ParlorWave Radio may DJ on: Settings > Devices "
            "& services > ParlorWave Radio > Configure > DJ-enabled outputs"))
    elif selected and selected not in enabled:
        out.append(_config.problem(
            "output_selection_not_enabled", _config.SEVERITY_WARNING, "outputs",
            f"the selected output {selected} is not DJ-enabled; playback on "
            "it is not followed and no break can air there",
            "select a DJ-enabled output, or add it to the DJ-enabled outputs"))
    elif len(enabled) > 1 and not selected:
        out.append(_config.problem(
            "output_selection_missing", _config.SEVERITY_WARNING, "outputs",
            f"{len(enabled)} outputs are DJ-enabled but none is selected as "
            "the output; until one is selected (or a station is started) no "
            "playback is followed",
            "pick the Output in ParlorWave Radio, or DJ-enable one output only"))
    return out


def observer_problems(cfg: _config.Config,
                      status: dict[str, Any]) -> list[dict[str, Any]]:
    """Configuration-level observer problems live in config.feature_problems
    (invalid mode, explicitly disabled). This is the RUNTIME check: in the
    product configuration (authoritative) the observer must be running."""
    if not cfg.observer_mode_valid or cfg.observer_mode != _config.OBSERVER_AUTHORITATIVE:
        return []
    if (status or {}).get("running"):
        return []
    return [_config.problem(
        "observer_not_running", _config.SEVERITY_ERROR, "observer",
        "the Music Assistant observer is configured authoritative but is not "
        "running; playback is not followed and no break can be scheduled",
        f"check the service log; {_config.ENV_OBSERVER_QUEUE} must be "
        f"'{_config.OBSERVER_QUEUE_AUTO}' (default) or a player id")]


def observer_summary(cfg: _config.Config, status: dict[str, Any]) -> dict[str, Any]:
    """Bounded observer facts for the status surfaces -- no rings, no
    counters beyond what a UI needs to say 'following X' or 'stopped'."""
    st = status or {}
    return {"configured_mode": cfg.observer_mode,
            "active_mode": st.get("active_mode") or _config.OBSERVER_OFF,
            "queue_mode": cfg.observer["queue_mode"],
            "running": bool(st.get("running")),
            "feeding": bool(st.get("feeding")),
            "queue_id": st.get("queue_id"),
            "target_reason": st.get("target_reason"),
            "last_error": st.get("last_error")}


def _speech_route() -> dict[str, Any] | None:
    """What the router would use for the stored selection. Never raises."""
    try:
        import tts_router
        return tts_router.decide()
    except Exception:  # noqa: BLE001 - a settings failure is not a health crash
        return None


def ha_tts_problems(cfg: _config.Config, route: dict[str, Any] | None,
                    ha_probe: dict[str, Any] | None,
                    entity_probe_fn) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(problems, selected-speech summary). Home Assistant unreachable or
    refusing the token is an ERROR when a Home Assistant voice is the
    selected route and a warning otherwise; a selected entity that is
    missing or unavailable is an error. Nothing here logs or returns the
    token."""
    import ha_tts
    # The stored selection, not only the route: with Home Assistant
    # unconfigured the router keeps the old route, but the customer's
    # choice is still a Home Assistant voice and must be reported as such.
    try:
        import tts_router
        selected_ref = str(tts_router._refs().get(tts_router.HA_REF) or "")
    except Exception:  # noqa: BLE001
        selected_ref = ""
    selected_ha = bool((route and route.get("provider") == "home_assistant")
                       or (selected_ref and not cfg.ha_configured))
    entity_id = (str(route.get("voice") or "") if route and route.get("provider") == "home_assistant"
                 else selected_ref) if selected_ha else ""
    summary: dict[str, Any] = {
        "provider": (route or {}).get("provider") or None,
        "entity_id": entity_id or None, "name": None, "state": None}
    problems: list[dict[str, Any]] = []
    if not cfg.ha_configured:
        if selected_ha:
            problems.append(_config.problem(
                "ha_unconfigured", _config.SEVERITY_ERROR, "tts",
                f"the selected voice speaks through Home Assistant ({entity_id}) "
                "but Home Assistant is not configured",
                f"set {_config.ENV_HA_URL} and {_config.ENV_HA_TOKEN}, or run as the add-on"))
            summary["state"] = "unconfigured"
        return problems, summary
    if not (ha_probe or {}).get("ok"):
        kind = (ha_probe or {}).get("kind")
        sev = _config.SEVERITY_ERROR if selected_ha else _config.SEVERITY_WARNING
        if kind == ha_tts.KIND_AUTH:
            problems.append(_config.problem(
                "ha_auth_rejected", sev, "home_assistant",
                "Home Assistant rejected the configured token",
                f"check {_config.ENV_HA_TOKEN} (or the add-on's Supervisor access)"))
            summary["state"] = "auth_rejected" if selected_ha else None
        else:
            problems.append(_config.problem(
                "ha_unreachable", sev, "home_assistant",
                f"Home Assistant at {cfg.ha_url} did not answer: "
                f"{(ha_probe or {}).get('error') or 'no response'}",
                f"check {_config.ENV_HA_URL} and that Home Assistant is running"))
            summary["state"] = "unreachable" if selected_ha else None
        return problems, summary
    if selected_ha:
        try:
            ent = entity_probe_fn(entity_id)
            summary["name"] = ent.get("name")
            if ent.get("available"):
                summary["state"] = "healthy"
            else:
                summary["state"] = "unavailable"
                problems.append(_config.problem(
                    "ha_tts_entity_unavailable", _config.SEVERITY_ERROR, "tts",
                    f"the selected Home Assistant voice {entity_id} is unavailable",
                    "check that its integration (Piper, Cloud, ...) is running"))
        except ha_tts.HAError as exc:
            if exc.kind == ha_tts.KIND_NOT_FOUND:
                summary["state"] = "missing"
                problems.append(_config.problem(
                    "ha_tts_entity_missing", _config.SEVERITY_ERROR, "tts",
                    f"the selected Home Assistant voice {entity_id} does not exist",
                    "pick a voice that exists in Home Assistant (tts.* entities)"))
            else:
                summary["state"] = "unreachable"
                problems.append(_config.problem(
                    "ha_unreachable", _config.SEVERITY_ERROR, "home_assistant",
                    f"Home Assistant could not report {entity_id}: {exc}",
                    f"check {_config.ENV_HA_URL} and that Home Assistant is running"))
    return problems, summary


def knowledge_block() -> dict[str, Any]:
    """The shipped music knowledge pack, as the health report sees it: is
    it here, is it the accepted candidate, and what is in it. A feature
    fact, never a path and never a fact from inside it."""
    import music_knowledge
    integrity = music_knowledge.integrity()
    meta = music_knowledge.metadata()
    return {
        "available": bool(music_knowledge.available()),
        "present": bool(integrity.get("present")),
        "verified": bool(integrity.get("matches")),
        "version": meta.get("version") or None,
        "candidate": meta.get("candidate") or None,
        "sources": list(meta.get("sources") or []),
        "artists": (meta.get("cohort") or {}).get("resolved_artists"),
        "facts": (meta.get("cohort") or {}).get("facts"),
    }


def dj_speech_problem() -> dict[str, Any] | None:
    """A hand-damaged dj.speech_enabled: reported, never acted on. The DJ
    already speaks (the historical default) because only an explicit false
    silences it; this says why the stored value is being ignored. Never
    raises: a diagnosis never breaks the report."""
    try:
        import settings
        code = settings.dj_speech_setting_problem()
    except Exception:  # noqa: BLE001
        return None
    if not code:
        return None
    return _config.problem(
        code, _config.SEVERITY_WARNING, "dj",
        "the saved Speak DJ breaks setting is not on or off, so it is being "
        "read as on and DJ breaks are spoken as before",
        "switch Speak DJ breaks off and on again to save a valid value")


def knowledge_problem() -> dict[str, Any] | None:
    """A pack that is present but is not the candidate this build was
    tested against is a WARNING: grounded music facts are optional
    enrichment, so the station keeps playing and keeps talking -- it just
    stops offering library knowledge. It is never fatal and never an
    error, because nothing about playback, queues or speech depends on
    it. It is also never silent: an unverified pack that quietly did
    nothing would look exactly like a station with nothing to say."""
    import music_knowledge
    integrity = music_knowledge.integrity()
    if integrity.get("matches"):
        return None
    if not integrity.get("present"):
        # No pack at all is a FEATURE FACT (knowledge_block says
        # present=false), not a fault: the public build deliberately
        # ships without one, so every public install would otherwise
        # carry a permanent warning whose advice ("reinstall") can never
        # clear it. Only a pack that IS here and is not the accepted
        # candidate is worth a word.
        return None
    return _config.problem(
        "music_knowledge_pack_unverified", _config.SEVERITY_WARNING,
        "music_knowledge",
        "the music knowledge pack is not the accepted candidate this build "
        "was tested against (checksum mismatch); it will not be used and "
        "breaks are written without grounded library facts",
        "reinstall or update the add-on")


def _dj_settings() -> dict[str, Any]:
    try:
        import settings as _settings
        return _settings.get_section("dj") or {}
    except Exception:  # noqa: BLE001 - a settings failure is not a health crash
        return {}


def _observer_status() -> dict[str, Any]:
    try:
        import ma_observer
        return ma_observer.status()
    except Exception:  # noqa: BLE001
        return {}


def report(cfg: _config.Config | None = None,
           probe_fn=None, llm_probe_fn=None, ha_probe_fn=None,
           ha_entity_fn=None, dj_settings_fn=None,
           observer_status_fn=None, catalog_count_fn=None) -> dict[str, Any]:
    """The whole picture, one bounded call."""
    import catalog_status
    import ha_tts
    import ma_queue
    import providers
    cfg = cfg or _config.current()
    dj = (dj_settings_fn or _dj_settings)()
    obs = (observer_status_fn or _observer_status)()
    # No Music Assistant token yet (the add-on's first-run state): there is
    # nothing to probe and nothing to learn from probing -- config already
    # reports the missing token, and a doomed call would only add a timeout
    # to every health read.
    token_missing = not cfg.ma_token()
    probe = {"ok": False, "url": cfg.ma_url, "http_status": None, "error": "",
             "players": 0, "token_present": False} if token_missing \
        else (probe_fn or ma_queue.probe)()
    writer = effective_writer(cfg)
    llm_probe = None
    if writer["mode"] == "llm":
        # Probe the endpoint the WRITER would use, not the one the
        # environment happens to name: a settings-configured model that is
        # down must be reported down, and an environment variable left
        # over from a previous setup must not be probed on its behalf.
        llm_probe = (llm_probe_fn or providers.ollama_probe)(
            url=writer["endpoint"])
    ha_probe = (ha_probe_fn or ha_tts.probe)() if cfg.ha_configured else None
    route = _speech_route()
    fatal = _config.required_problems(cfg)
    problems: list[dict[str, Any]] = list(fatal)
    ma = None if token_missing else ma_problem(probe)
    if ma and ma["code"] not in {p["code"] for p in problems}:
        problems.append(ma)
    wp = writer_problem(writer, llm_probe)
    if wp:
        problems.append(wp)
    wsp = writer_settings_problem(writer)
    if wsp:
        problems.append(wsp)
    tts_block = effective_tts(cfg)
    ha_probs, speech = ha_tts_problems(cfg, route, ha_probe,
                                       ha_entity_fn or ha_tts.entity_state)
    problems.extend(ha_probs)
    problems.extend(selected_provider_problems(cfg, route, tts=tts_block))
    problems.extend(outputs_problems(dj))
    problems.extend(observer_problems(cfg, obs))
    problems.extend(autoplay_watch.problems())
    problems.extend(_config.feature_problems(cfg))
    kp = knowledge_problem()
    if kp:
        problems.append(kp)
    sp = dj_speech_problem()
    if sp:
        problems.append(sp)
    try:
        import announcements
        problems.extend(announcements.health_problems())
    except Exception:  # noqa: BLE001 - a diagnosis never breaks the report
        pass
    if tts_block["configured"]:
        # config.feature_problems reads the environment and is deliberately
        # pure, so it cannot see a voice engine that lives in settings. An
        # installation with ONLY that engine is configured, and must not be
        # told that no speech provider is.
        problems = [p for p in problems if p["code"] != "tts_unconfigured"]
    outputs = (dj or {}).get("outputs") or {}
    errors = [p for p in problems
              if p["severity"] in (_config.SEVERITY_FATAL, _config.SEVERITY_ERROR)]
    return {
        "ok": not errors,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instance_id": cfg.instance_id or None,
        "fatal": fatal,
        "problems": problems,
        "features": {
            "music_assistant": {
                "configured": bool(cfg.ma_url) and not token_missing,
                "reachable": bool(probe.get("ok")),
                "url": cfg.ma_url or None,
                # Whether a token EXISTS, never the token. A setup surface
                # has to be able to say "an access token is present" or
                # "one is missing" without ever holding the value, and
                # `configured` alone cannot tell the two apart from the
                # address being unset.
                "token_present": bool(probe.get("token_present")),
                "players": probe.get("players", 0),
            },
            "writer": writer_block(writer, llm_probe),
            "tts": {**tts_block, "selected": speech},
            "home_assistant": {"configured": bool(cfg.ha_configured),
                               "mode": cfg.ha_mode or None,
                               "url": cfg.ha_url or None,
                               "reachable": (bool(ha_probe.get("ok"))
                                             if ha_probe else None)},
            "audio": {"advertise_url": cfg.advertise_url or None,
                      "break_audio_base": cfg.audio_url_base or None},
            # Whether this install can PROGRAMME music itself, as opposed
            # to following what Music Assistant plays. Deliberately a
            # feature fact and NOT a problem: an install with no local
            # catalog is a supported install, not a broken one -- see
            # catalog_status. No path, no filename, no source.
            "catalog": catalog_status.report(catalog_count_fn),
            # The shipped grounded-facts pack. Like `catalog`, a feature
            # fact: an install whose pack is absent is a working install
            # with one fewer thing to say, not a broken one.
            "music_knowledge": knowledge_block(),
            "outputs": {"dj_enabled": bool((dj or {}).get("enabled")),
                        "enabled": list(outputs.get("enabled") or []),
                        "selected": outputs.get("default_request_output") or None},
            "observer": observer_summary(cfg, obs),
            # The Station Programmer (Mode 2): is a station on air, where,
            # and is anything wrong with its runway. A feature fact from
            # local state only -- no Music Assistant call is made for it.
            "station_programmer": station_programmer_block(),
        },
    }


def station_programmer_block() -> dict[str, Any]:
    """Compact Mode 2 summary for the health document. Fenced: an
    unreadable station store reports itself, never breaks health."""
    try:
        import station_live
        s = station_live.status()
    except Exception:  # noqa: BLE001 - health must never be the thing that breaks
        return {"active": None, "state": "unknown", "error": "unreadable"}
    st = s.get("station") or {}
    runway = st.get("runway") or {}
    last = (st.get("refill") or {}).get("last") or {}
    return {"active": bool(s.get("active")), "state": s.get("state"),
            "label": st.get("label") or None,
            "target": (st.get("target") or {}).get("queue_id") or None,
            "owned_ahead": runway.get("owned"),
            "music_ahead": runway.get("playable"),
            "checked_at": runway.get("at"),
            "refills": (st.get("refill") or {}).get("count"),
            "last_refill_at": last.get("at"),
            "last_refill_added": last.get("appended"),
            "last_refill_reason": last.get("reason") or None,
            "problem": st.get("problem") or None}
