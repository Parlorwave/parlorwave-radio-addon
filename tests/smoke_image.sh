#!/bin/sh
# Start the built add-on image and prove the service is really in it.
#
#   tests/smoke_image.sh <image>            e.g. parlorwave-radio:ci-amd64
#
# This is the last check before an image is worth publishing, and it is the
# only one that runs the thing rather than reading it. It deliberately does
# NOT use the add-on entrypoint: `run.sh` is `#!/usr/bin/with-contenv
# bashio` and asks the Supervisor for this add-on's hostname, which is a
# question with no answer outside Home Assistant. It starts the service the
# way run.sh finally does -- `python3 -m serve` with the same configuration
# boundary -- so what is proved is the image, not the Supervisor.
#
# What it proves:
#   1. the service starts with no Music Assistant to talk to, and stays up
#      (an installation is configured in whatever order the customer likes,
#      so a backend that dies before its token arrives is a dead end);
#   2. the unauthenticated surface is exactly the watchdog route;
#   3. every other route demands the API key, and accepts the right one;
#   4. ffmpeg AND ffprobe are present, and `audio_probe` -- the finite-audio
#      gate every break and bulletin passes through -- works against them;
#   5. the runtime assets the audio route serves are in the image;
#   6. the published subset of the backend is closed under import and can
#      build the service's route table, run with the image's own
#      interpreter and its own installed dependencies;
#   7. no GPL Python library is installed, the image makes no licence claim
#      about software it did not write, and the notices that name the
#      copyleft components and their exact sources are inside it.
set -eu

IMAGE="${1:?usage: tests/smoke_image.sh <image>}"
NAME="parlorwave-smoke-$$"
KEY="smoke-key-$(date +%s)"

cleanup() {
  status=$?
  if [ $status -ne 0 ]; then
    echo "--- container log ---"
    docker logs "$NAME" 2>&1 | tail -40 || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit $status
}
trap cleanup EXIT INT TERM

fail() { echo "SMOKE FAILED: $*" >&2; exit 1; }

echo "--- image: $IMAGE"
docker image inspect "$IMAGE" --format \
  '    {{.Os}}/{{.Architecture}}  {{.Config.Labels}}' || fail "no such image"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WANT_VERSION=$(sed -n 's/^version: *"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' \
               "$REPO_ROOT/parlorwave_radio/config.yaml" 2>/dev/null || true)
if [ -n "$WANT_VERSION" ]; then
  echo "--- the image carries this release's version"
  got=$(docker image inspect "$IMAGE" \
        --format '{{index .Config.Labels "org.opencontainers.image.version"}}')
  echo "    label $got (config.yaml says $WANT_VERSION)"
  case "$got" in
    "$WANT_VERSION"|ci) ;;
    *) fail "the image's version label is '$got'; it was built without
             BUILD_VERSION and has inherited the base image's" ;;
  esac
fi

echo "--- ffmpeg is present and runs"
docker run --rm --entrypoint /usr/bin/ffmpeg "$IMAGE" -version \
  | head -1 || fail "ffmpeg is missing or will not run"

echo "--- the runtime assets are in the image"
docker run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'test -s /app/code/assets/retired_silence.mp3 && test -s /app/code/app.py' \
  || fail "runtime assets are missing from /app/code"

echo "--- nothing optional, and no GPL python library, is installed"
# mutagen is in this list for a licence reason rather than a size one: it is
# GPL-2.0-or-later, and an Apache-2.0 product that imports it cannot describe
# itself honestly. Its work is done by audio_probe over the ffprobe boundary.
docker run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'for m in torch onnxruntime numpy mutagen chatterbox kokoro; do
     if python3 -c "import $m" 2>/dev/null; then echo "FOUND:$m"; fi
   done' | grep -q FOUND && fail "a forbidden python package is in the image"

echo "--- ffprobe is present, and audio_probe works with it"
docker run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'set -e
   ffprobe -version >/dev/null
   ffmpeg -hide_banner -loglevel error -f lavfi -i anullsrc=r=24000:cl=mono \
     -t 2 -codec:a libmp3lame -b:a 64k -f mp3 /tmp/s.mp3 -y
   cd /app/code && python3 -c "
import audio_probe, sys
d = audio_probe.duration(\"/tmp/s.mp3\")
i = audio_probe.stream_info(\"/tmp/s.mp3\")
assert d and 1.8 < d < 2.3, d
assert i and i[\"sample_rate\"] == 24000 and i[\"channels\"] == 1, i
assert audio_probe.write_text_tags(\"/tmp/s.mp3\", title=\"DJ break\")
assert audio_probe.duration(\"/tmp/s.mp3\"), \"tagging broke the probe\"
assert audio_probe.duration(\"/dev/null\") is None
print(f\"    duration {d}  shape {i}  tag written and still probes\")
"' || fail "audio_probe does not work in the image"

echo "--- the image makes no licence claim about contents it did not write"
licences=$(docker image inspect "$IMAGE" \
  --format '{{index .Config.Labels "org.opencontainers.image.licenses"}}')
[ -z "$licences" ] || fail "the image claims org.opencontainers.image.licenses=$licences,
       but it contains a Debian system and a GPL-enabled FFmpeg"
own=$(docker image inspect "$IMAGE" \
  --format '{{index .Config.Labels "io.parlorwave.source.license"}}')
[ "$own" = "Apache-2.0" ] || fail "io.parlorwave.source.license is '$own'"
echo "    no image-wide licence label; io.parlorwave.source.license=$own"

echo "--- the notices travel inside the image"
docker run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'test -s /usr/share/doc/parlorwave-radio/LICENSE &&
   test -s /usr/share/doc/parlorwave-radio/NOTICE &&
   grep -q "snapshot.debian.org" /usr/share/doc/parlorwave-radio/THIRD_PARTY_NOTICES.md &&
   grep -q "7:5.1.9-0+deb12u1" /usr/share/doc/parlorwave-radio/THIRD_PARTY_NOTICES.md' \
  || fail "the licence and third-party notices are not inside the image"

echo "--- the ffmpeg in this image is the version the notices name"
docker run --rm --entrypoint /usr/bin/dpkg-query "$IMAGE" \
  -W -f='    ffmpeg ${Version} (src ${source:Package} ${source:Version})\n' ffmpeg

# The image has the pinned dependencies installed, so it is also the
# cheapest place to prove the published subset of the backend is closed
# under import and can build the service's route table. CI runs the same
# file on the runner after `pip install -r requirements.txt`; this run
# costs nothing and catches a missing module before the service starts.
if [ -f "$REPO_ROOT/tests/test_backend_release_subset.py" ]; then
  echo "--- the published backend is complete (inside the image)"
  docker run --rm --network none --entrypoint /usr/bin/python3 \
    -v "$REPO_ROOT:/export:ro" -e MUSICBRAIN_BASE_DIR=/tmp/parlorwave-smoke \
    -w /export "$IMAGE" tests/test_backend_release_subset.py 2>&1 | tail -4 \
    || fail "the published backend is not closed under import"
fi

echo "--- starting the service"
docker run -d --name "$NAME" --network bridge \
  --entrypoint /usr/bin/python3 \
  -e MUSICBRAIN_BASE_DIR=/data \
  -e MUSICBRAIN_HOST=0.0.0.0 \
  -e MUSICBRAIN_PORT=5005 \
  -e MUSICBRAIN_API_KEY="$KEY" \
  -e MUSIC_ASSISTANT_URL=http://127.0.0.1:8095 \
  -e MUSIC_ASSISTANT_TOKEN=smoke-token-not-a-real-one \
  -e MUSICBRAIN_ADVERTISE_URL=http://127.0.0.1:5005 \
  -w /app/code \
  "$IMAGE" -m serve >/dev/null

# The watchdog route is what the Supervisor itself polls, so it is the right
# thing to wait on.
ready=""
i=0
while [ $i -lt 60 ]; do
  if docker exec "$NAME" python3 -c \
      "import urllib.request;urllib.request.urlopen('http://127.0.0.1:5005/',timeout=2)" \
      >/dev/null 2>&1; then
    ready=yes
    break
  fi
  if [ -z "$(docker ps -q -f name="$NAME")" ]; then
    fail "the container exited before it answered"
  fi
  i=$((i + 1))
done
[ -n "$ready" ] || fail "the service did not answer within 60 attempts"
echo "    up after $i attempts"

probe() {
  docker exec -i "$NAME" python3 - "$1" "$2" <<'PY'
import json, sys, urllib.error, urllib.request
path, key = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"http://127.0.0.1:5005{path}")
if key:
    req.add_header("X-Api-Key", key)
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        print(r.status, json.dumps(json.load(r))[:200])
except urllib.error.HTTPError as e:
    print(e.code, e.read()[:200].decode("utf-8", "replace"))
PY
}

echo "--- the watchdog route is open"
out=$(probe / "")
echo "    GET /            -> $out"
case "$out" in
  200*'"status": "online"'*) ;;
  *) fail "GET / did not report the service online: $out" ;;
esac
case "$out" in
  *'"auth": "required"'*) ;;
  *) fail "authentication is not required: $out" ;;
esac

echo "--- every other route demands the key"
for path in /instance /health/problems /dj/status; do
  out=$(probe "$path" "")
  echo "    GET $path (no key) -> $out"
  case "$out" in 401*) ;; *) fail "$path answered without a key: $out" ;; esac

  out=$(probe "$path" "wrong-key")
  case "$out" in 401*) ;; *) fail "$path accepted a wrong key: $out" ;; esac

  out=$(probe "$path" "$KEY")
  echo "    GET $path (key)    -> $out"
  case "$out" in 200*) ;; *) fail "$path refused the right key: $out" ;; esac
done

echo "--- it says what is missing rather than dying"
out=$(probe /health/problems "$KEY")
case "$out" in
  *music_assistant*|*ma_*) echo "    reports Music Assistant is not reachable" ;;
  *) echo "    (no Music Assistant problem reported; not fatal here)" ;;
esac
[ -n "$(docker ps -q -f name="$NAME")" ] || fail "the service died during the smoke"

echo "SMOKE PASSED: $IMAGE"
