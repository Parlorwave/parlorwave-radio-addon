# https://developers.home-assistant.io/docs/apps/configuration#app-dockerfile
# One Debian 12 base for both architectures (a multi-arch manifest: the
# Supervisor / docker `--platform` picks amd64 or arm64); its python3 is
# 3.11, the runtime the backend is tested on. build.yaml is deprecated by
# the Supervisor (2026.09 warns), so the base lives here. It is pinned BY
# DIGEST (Home Assistant base-debian 2026.10.3): the `bookworm` tag moves,
# and the Debian snapshot below fixes every package, FFmpeg's version among them.
ARG BUILD_FROM=ghcr.io/home-assistant/base-debian:bookworm@sha256:ebb912a11616e3eda2d31204803587276e70bbb4e7f02a6c92dba5f43256b48d
FROM ${BUILD_FROM}

# The release, as a build argument, because an image that does not carry its
# own version carries the BASE image's -- the label is inherited, so it is
# wrong rather than absent, which is worse. `BUILD_VERSION` is the name Home
# Assistant's own builder passes; scripts/build_addon_image.sh and the
# add-on repository's publish workflow both read it out of config.yaml, so
# the label, the image tag and `version:` are one string.
ARG BUILD_VERSION=0.0.0-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Runtime: Debian 12's python3 (3.11) and ffmpeg (finite MP3 breaks).
# Nothing else: no torch, no Kokoro, no Chatterbox, no Ollama, no weights.
# From ONE Debian archive SNAPSHOT, never from whatever bookworm holds on the
# day of the build: with the base digest and requirements.txt, every package
# in the image is fixed. Signatures are still checked. The snapshot is newer
# than the base's, so the base's own packages are upgraded to it first (the
# security updates for libexpat1, OpenSSL, libbrotli1 and tzdata); `upgrade`
# installs and removes nothing, and both archives resolve to the same instant.
ARG DEBIAN_SNAPSHOT=20261028T143257Z
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: bookworm bookworm-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: bookworm-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' \
        "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The backend (staged next to this Dockerfile by the release exporter:
# the add-on build context must contain everything it copies).
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --no-deps -r /tmp/requirements.txt

COPY backend /app/code
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# The licence and the third-party notices travel INSIDE the image. Somebody
# who pulls it has not necessarily seen the repository, and the FFmpeg and
# Debian components in here are copyleft: the instructions for obtaining
# their corresponding source have to be somewhere the recipient actually
# has. `dpkg-query` and /usr/share/doc/<pkg>/copyright are the rest of that
# answer and are already present in every layer below this one.
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /usr/share/doc/parlorwave-radio/

# The running service reports its own release (GET /instance -> version),
# so the integration can tell an add-on left on another release from a
# matching one. Set this late: a version change rebuilds one cheap layer,
# not the apt and pip layers above.
ENV MUSICBRAIN_VERSION="${BUILD_VERSION}"

WORKDIR /app/code
CMD ["/run.sh"]

# NO `org.opencontainers.image.licenses`, AND THAT IS DELIBERATE.
#
# That annotation describes the licence of the software CONTAINED in the
# image, and this image contains a Debian system, FFmpeg built with
# --enable-gpl, and a few hundred packages besides -- GPL-2+, GPL-3+,
# LGPL-2.1+, MIT, BSD, public domain and more. "Apache-2.0" there would be
# a false statement about all of it, and no single SPDX expression
# truthfully describes the aggregate without inventing one nobody has
# verified. The annotation is optional, so it is omitted rather than
# guessed at.
#
# What IS true is stated instead, scoped so it cannot be mistaken for a
# claim about the whole image: ParlorWave's OWN source is Apache-2.0, and
# the notices covering everything else are inside the image at a path a
# recipient can read.
LABEL \
    org.opencontainers.image.title="ParlorWave Radio" \
    org.opencontainers.image.description="The DJ for your Music Assistant stations." \
    org.opencontainers.image.source="https://github.com/parlorwave/parlorwave-radio-addon" \
    org.opencontainers.image.url="https://github.com/parlorwave/parlorwave-radio" \
    org.opencontainers.image.version="${BUILD_VERSION}" \
    io.parlorwave.source.license="Apache-2.0" \
    io.parlorwave.third-party-notices="/usr/share/doc/parlorwave-radio/THIRD_PARTY_NOTICES.md"
