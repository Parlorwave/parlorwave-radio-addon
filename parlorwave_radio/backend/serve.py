"""Start the backend: `python -m serve`.

Checks the required configuration BEFORE the web server exists, so a
misconfigured service exits with a short, complete list of what to set --
not a framework traceback. Exit status 2 = configuration; the app itself
repeats the check in its lifespan as a second line of defence.
"""

from __future__ import annotations

import os
import sys

import config


def main() -> int:
    cfg = config.current()
    fatal = config.required_problems(cfg)
    if fatal:
        print("ParlorWave Radio backend cannot start -- required configuration "
              "is missing or invalid:", file=sys.stderr)
        for p in fatal:
            print(f"  - {p['summary']}\n      -> {p['action']}", file=sys.stderr)
        return 2
    if cfg.auth_disabled:
        print(f"WARNING: {config.ENV_AUTH_DISABLED}=1 -- API authentication is "
              "OFF; every endpoint is open. Development and tests only.",
              file=sys.stderr)
    import uvicorn
    # No per-request access log. The integration polls this service every
    # ten seconds (every two while a break is imminent), seven routes a
    # cycle: with uvicorn's default access log that is ~60,000 lines a day
    # of "GET /dj/status 200" written into the host's add-on log for a
    # service whose own operational lines fit on one screen. Errors, the
    # startup lines and everything on the musicbrain logger are unaffected
    # (logging_setup.py). Storage endurance: the SD-card write budget.
    uvicorn.run("app:app",
                host=os.environ.get("MUSICBRAIN_HOST", "0.0.0.0"),
                port=int(os.environ.get("MUSICBRAIN_PORT", "5005")),
                workers=1, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
