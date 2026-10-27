#!/usr/bin/env python3

import json
import os
from typing import Any
from urllib import error, request

import config as _config
from paths import BASE_DIR

CONFIG_PATH = BASE_DIR / "lidarr_config.json"
LIDARR_URL = _config.current().lidarr_url          # "" = not configured
LIDARR_API_KEY = os.environ.get("LIDARR_API_KEY")


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def get_lidarr_settings() -> dict[str, Any]:
    config = _load_config()
    url = config.get("url") or os.environ.get("LIDARR_URL") or LIDARR_URL
    api_key = config.get("api_key") or os.environ.get("LIDARR_API_KEY") or LIDARR_API_KEY
    return {"url": (url or "").rstrip("/"), "api_key": api_key}


def _request(path: str, *, timeout: int = 15) -> Any:
    settings = get_lidarr_settings()
    if not settings["url"]:
        raise RuntimeError("Lidarr is not configured (LIDARR_URL unset)")
    url = f"{settings['url']}{path}"
    headers = {}
    if settings.get("api_key"):
        headers["X-Api-Key"] = settings["api_key"]
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except error.HTTPError as exc:
        raise RuntimeError(f"Lidarr API error {exc.code}: {exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Lidarr at {url}: {exc.reason}") from exc


def get_lidarr_status() -> dict[str, Any]:
    settings = get_lidarr_settings()
    if not settings["url"]:
        return {"ok": False, "configured": False, "url": "",
                "error": "Lidarr is not configured (LIDARR_URL unset)"}
    try:
        data = _request("/api/v1/system/status")
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "url": settings["url"],
            "configured": bool(settings.get("api_key")),
        }

    return {
        "ok": True,
        "url": settings["url"],
        "configured": bool(settings.get("api_key")),
        "version": data.get("version") if isinstance(data, dict) else None,
        "app_name": data.get("appName") if isinstance(data, dict) else None,
    }
