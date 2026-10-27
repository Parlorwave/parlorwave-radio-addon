#!/usr/bin/with-contenv bashio
# Parlorwave Radio add-on entrypoint. Turns add-on options and the
# Supervisor's facts into the backend's configuration boundary (config.py),
# then hands over to the Python bootstrap (key persistence, discovery,
# serve). Nothing here is installation-specific.
#
# with-contenv matters: the base image boots through s6-overlay, which
# starts the CMD with a scrubbed environment; only with-contenv restores
# the container environment (SUPERVISOR_TOKEN included). Proven on a real
# Supervisor: without it every Supervisor call is "invalid token".
set -e

export MUSICBRAIN_BASE_DIR=/data
export MUSICBRAIN_HOST=0.0.0.0
export MUSICBRAIN_PORT=5005

# Music Assistant: the add-on options (URL + API token). The default URL is
# the Music Assistant add-on on this same host: it runs on the host network,
# and the internal name `homeassistant` is the host on the add-on network
# (verified on HA OS). A missing token is
# reported by the backend at startup and on /health/problems, never guessed.
export MUSIC_ASSISTANT_URL="$(bashio::config 'music_assistant_url' 'http://homeassistant:8095')"
export MUSIC_ASSISTANT_TOKEN="$(bashio::config 'music_assistant_token' '')"

# The add-on always runs the standard DJ writer. The backend's experimental
# writer (DJ_SMART_WRITER) is a development control, not an add-on option:
# whatever the image environment carried, it is unset here.
unset DJ_SMART_WRITER DJ_SMART_WRITER_P2_MODEL

# This add-on as the internal network sees it, straight from the
# Supervisor's own record (bashio reads /addons/self/info) -- never
# constructed from the slug, which carries a repository prefix once the
# add-on is installed from a repository rather than locally.
#
# These are FALLBACKS: the Python bootstrap asks the Supervisor itself and
# derives the advertised audio URL from the hostname, because the container
# address changes between runs while the hostname is re-registered against
# the current container. Nothing here sets MUSICBRAIN_ADVERTISE_URL, so the
# bootstrap's hostname-based value stands (an explicit override still wins).
ADDON_HOSTNAME="$(bashio::addon.hostname)"
ADDON_IP="$(bashio::addon.ip_address)"
export PARLORWAVE_ADDON_HOSTNAME="${ADDON_HOSTNAME}"
export PARLORWAVE_ADDON_IP="${ADDON_IP}"

bashio::log.info "Parlorwave Radio add-on: hostname ${ADDON_HOSTNAME}, ip ${ADDON_IP}"
if [ -z "${MUSIC_ASSISTANT_TOKEN}" ]; then
  bashio::log.warning "music_assistant_token is not set: create an access token in Music Assistant (Profile > Access tokens) and enter it in this add-on's Configuration tab"
fi

cd /app/code
exec python3 -m addon_bootstrap
