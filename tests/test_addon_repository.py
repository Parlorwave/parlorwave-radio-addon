"""This is a valid Home Assistant add-on repository, and it publishes the
image it says it publishes.

    python3 tests/test_addon_repository.py

No Supervisor is needed and none is pretended. Everything here is checked
against the documented add-on configuration contract and against the
Supervisor's own validation rules:

  * `arch` must be drawn from ARCH_ALL -- `aarch64` and `amd64`. `armhf`,
    `armv7` and `i386` are ARCH_DEPRECATED and the Supervisor logs a warning
    for them (supervisor/const.py, supervisor/apps/validate.py).
  * `image` is passed through `str.format(arch=...)`, so `{arch}` is the
    only placeholder that exists and a name without one is used verbatim
    (supervisor/apps/model.py `_image`). It must carry NO tag: the tag is
    `version` (supervisor/validate.py `docker_image`).
  * The image is pulled with an explicit `platform=`, from MAP_ARCH
    {aarch64: linux/arm64, amd64: linux/amd64}, so ONE multi-architecture
    manifest resolves correctly on the customer's machine
    (supervisor/docker/interface.py).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "parlorwave_radio"
CONFIG = ADDON / "config.yaml"

ARCH_ALL = {"aarch64", "amd64"}
ARCH_DEPRECATED = {"armhf", "armv7", "i386"}

#: supervisor/validate.py: each path component of an image name.
IMAGE_COMPONENT = re.compile(r"^[a-z0-9{][a-z0-9.\-_{}]*$")
#: supervisor/validate.py: RE_SLUG.
SLUG = re.compile(r"^[-_.A-Za-z0-9]+$")


def load(text: str) -> dict:
    """Top-level `key: value` pairs. There is no PyYAML here on purpose:
    this suite installs nothing, so it reads only the shape it needs and
    `load_lists` adds the two sequences an add-on config has."""
    out: dict = {}
    for raw in text.splitlines():
        if not raw or raw[0] in " #-":
            continue
        key, sep, val = raw.partition(":")
        if not sep or " " in key.strip():
            continue
        val = val.split(" #", 1)[0].strip()
        if val:
            out[key.strip()] = scalar(val)
    return out


def scalar(v: str):
    v = v.strip()
    if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    if v == "null":
        return None
    return v


def load_lists(text: str) -> dict:
    """The reader above, plus every top-level sequence -- `arch` and
    `discovery` are the two this add-on has."""
    data = load(text)
    for block in re.finditer(r"^([a-z_]+):[ \t]*$\n((?:[ \t]+-[ \t]+.*\n?)+)",
                             text, re.M):
        data[block.group(1)] = [scalar(line.strip()[2:])
                                for line in block.group(2).splitlines()
                                if line.strip()]
    return data


class RepositoryLayout(unittest.TestCase):
    def test_repository_yaml_is_at_the_root(self):
        data = load((ROOT / "repository.yaml").read_text())
        self.assertEqual(data["name"], "ParlorWave Radio")
        self.assertEqual(data["url"],
                         "https://github.com/parlorwave/parlorwave-radio-addon")
        self.assertIn("maintainer", data)

    def test_the_addon_directory_is_complete(self):
        for name in ("config.yaml", "Dockerfile", "run.sh", "README.md",
                     "CHANGELOG.md", "translations/en.yaml"):
            self.assertTrue((ADDON / name).is_file(), f"missing {name}")

    def test_there_is_exactly_one_addon(self):
        dirs = [p.name for p in ROOT.iterdir()
                if p.is_dir() and (p / "config.yaml").exists()]
        self.assertEqual(dirs, ["parlorwave_radio"], dirs)


class Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = CONFIG.read_text()
        cls.cfg = load_lists(cls.raw)

    def test_identity(self):
        self.assertEqual(self.cfg["name"], "ParlorWave Radio")
        self.assertEqual(self.cfg["slug"], "parlorwave_radio")
        self.assertRegex(self.cfg["slug"], SLUG)
        self.assertEqual(self.cfg["url"],
                         "https://github.com/parlorwave/parlorwave-radio-addon")
        self.assertTrue(self.cfg["description"])

    def test_version_is_a_usable_docker_tag(self):
        version = self.cfg["version"]
        self.assertRegex(version, r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
        heading = re.search(rf"^## {re.escape(version)}$",
                            (ADDON / "CHANGELOG.md").read_text(), re.M)
        self.assertTrue(heading, f"CHANGELOG has no '## {version}' heading")

    def test_architectures(self):
        arch = set(self.cfg["arch"])
        self.assertEqual(arch, ARCH_ALL, arch)
        self.assertFalse(arch & ARCH_DEPRECATED)

    def test_the_published_image(self):
        image = self.cfg["image"]
        self.assertEqual(image, "ghcr.io/parlorwave/parlorwave-radio-addon")
        registry, _, path = image.partition("/")
        self.assertEqual(registry, registry.lower())
        self.assertNotIn(":", path, "the tag is `version`, never part of `image`")
        for component in path.split("/"):
            self.assertRegex(component, IMAGE_COMPONENT, component)
        # One manifest for both architectures: no {arch} to substitute.
        self.assertNotIn("{arch}", image)

    def test_the_permission_set_is_the_minimum(self):
        self.assertTrue(self.cfg["homeassistant_api"])
        self.assertEqual(self.cfg["discovery"], ["parlorwave_radio"])
        for forbidden in ("host_network", "privileged", "docker_api",
                          "full_access", "host_pid", "hassio_role",
                          "auth_api", "devices", "usb", "uart", "video",
                          "gpio", "map"):
            self.assertNotIn(f"\n{forbidden}:", self.raw,
                             f"the add-on asks for {forbidden}")

    def test_no_host_port_is_published(self):
        """`ports: {5005/tcp: null}` means the Supervisor publishes nothing.
        Home Assistant and Music Assistant reach the service on the internal
        add-on network; a customer may map the port by hand for debugging."""
        ports = re.search(r"^ports:\n((?:[ \t]+.*\n)+)", self.raw, re.M)
        self.assertTrue(ports, "no ports: block")
        self.assertRegex(ports.group(1), r"5005/tcp:\s*null")

    def test_the_only_thing_a_customer_types_is_the_music_assistant_token(self):
        options = re.search(r"^options:\n((?:\s+.*\n)+)", self.raw, re.M)
        self.assertTrue(options)
        keys = set(re.findall(r"^\s+([a-z_]+):", options.group(1), re.M))
        self.assertEqual(keys, {"music_assistant_url", "music_assistant_token"})
        self.assertIn('music_assistant_token: "password?"', self.raw,
                      "the token must be optional and write-only in the UI")


class Dockerfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ADDON / "Dockerfile").read_text()
        cls.instructions = "\n".join(
            line for line in cls.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")).lower()

    def test_it_builds_on_the_home_assistant_base(self):
        self.assertRegex(self.text,
                         r"ARG BUILD_FROM=ghcr\.io/home-assistant/[\w-]*base-debian")
        self.assertIn("FROM ${BUILD_FROM}", self.text)

    def test_the_runtime_is_python_and_ffmpeg_and_nothing_else(self):
        install = re.search(r"apt-get install[^\n]*(?:\\\n[^\n]*)*", self.text)
        self.assertTrue(install)
        packages = set(re.findall(r"\b(python3|python3-pip|ffmpeg|[a-z0-9.+-]{3,})\b",
                                  install.group(0))) - {
            "apt-get", "install", "get", "recommends", "no-install-recommends"}
        self.assertLessEqual(packages, {"python3", "python3-pip", "ffmpeg", "y"},
                             f"unexpected apt packages: {packages}")

    def test_nothing_optional_is_installed(self):
        """The INSTRUCTIONS, not the comments. This Dockerfile says in prose
        that it installs no torch and no Kokoro, and a search over the whole
        file would fail on the very sentence that promises it."""
        for absent in ("torch", "kokoro", "chatterbox", "onnxruntime",
                       "espeak", "cuda", "nvidia", ".onnx", ".pt "):
            self.assertNotIn(absent, self.instructions, absent)

    def test_the_image_says_where_it_came_from_and_which_release_it_is(self):
        """An image with no version label does not lack one -- it INHERITS
        the base image's, which is wrong rather than absent. BUILD_VERSION
        is the argument Home Assistant's own builder passes, and every
        build path here passes the `version` from config.yaml."""
        self.assertIn(
            'org.opencontainers.image.source="https://github.com/parlorwave/'
            'parlorwave-radio-addon"', self.text)
        self.assertIn('org.opencontainers.image.version="${BUILD_VERSION}"', self.text)
        self.assertRegex(self.text, r"(?m)^ARG BUILD_VERSION=")

    def test_the_running_service_knows_its_release(self):
        """GET /instance reports the release from MUSICBRAIN_VERSION, which
        the image sets from the same BUILD_VERSION as its label -- after the
        apt and pip layers, so a version bump rebuilds one cheap layer."""
        self.assertRegex(self.text, r'(?m)^ENV MUSICBRAIN_VERSION="\$\{BUILD_VERSION\}"')
        self.assertGreater(self.text.index("ENV MUSICBRAIN_VERSION"),
                           self.text.index("pip3 install"))

    def test_the_image_makes_no_licence_claim_about_its_own_contents(self):
        """`org.opencontainers.image.licenses` describes the licence of the
        software CONTAINED in an image. This one contains a Debian system
        and a GPL-enabled FFmpeg, so `Apache-2.0` there would be false
        about nearly all of it, and no single SPDX expression truthfully
        describes the aggregate. The annotation is optional and is omitted.

        What is true is labelled instead, under names that cannot be read
        as a claim about the whole image."""
        self.assertNotIn("org.opencontainers.image.licenses", self.instructions,
                         "the image claims a licence for contents it did not write")
        self.assertIn('io.parlorwave.source.license="Apache-2.0"', self.text)
        self.assertIn("io.parlorwave.third-party-notices=", self.text)

    def test_the_notices_travel_inside_the_image(self):
        """Somebody who pulls an image has not necessarily seen this
        repository, and the copyleft components are exactly the ones whose
        source instructions must travel with the binary."""
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            self.assertTrue((ADDON / name).is_file(),
                            f"{name} is not in the build context")
        self.assertIn("COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md "
                      "/usr/share/doc/parlorwave-radio/", self.text)
        notices = (ADDON / "THIRD_PARTY_NOTICES.md").read_text()
        # the two things a recipient of the IMAGE alone cannot otherwise get
        self.assertIn("snapshot.debian.org", notices)
        self.assertIn("7:5.1.9-0+deb12u1", notices)

    def test_the_build_context_is_self_contained(self):
        """Every path this Dockerfile COPYs must exist inside the add-on
        directory. A docker build context is that directory and nothing
        else; a COPY naming anything outside it is a build that cannot
        succeed from a clean clone of this repository."""
        missing = []
        for match in re.finditer(r"^COPY\s+(?:--[^\s]+\s+)*(.+)$", self.text, re.M):
            parts = match.group(1).split()
            for source in parts[:-1]:
                if source.startswith("$") or source.startswith("/"):
                    continue
                if not (ADDON / source).exists():
                    missing.append(source)
        self.assertEqual(missing, [], f"COPY sources outside the context: {missing}")

    def test_no_build_yaml(self):
        """build.yaml is deprecated by the Supervisor; the base image lives
        in the Dockerfile's BUILD_FROM."""
        self.assertFalse((ADDON / "build.yaml").exists())


class Entrypoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ADDON / "run.sh").read_text()

    def test_with_contenv(self):
        """The base image boots through s6-overlay, which starts CMD with a
        scrubbed environment. Only `with-contenv` restores it, SUPERVISOR_TOKEN
        included -- without it every Supervisor call is 'invalid token'."""
        self.assertTrue(self.text.startswith("#!/usr/bin/with-contenv bashio"))

    def test_it_sets_the_configuration_boundary_and_nothing_else(self):
        exported = set(re.findall(r"^export ([A-Z_]+)=", self.text, re.M))
        self.assertEqual(exported, {
            "MUSICBRAIN_BASE_DIR", "MUSICBRAIN_HOST", "MUSICBRAIN_PORT",
            "MUSIC_ASSISTANT_URL", "MUSIC_ASSISTANT_TOKEN",
            "PARLORWAVE_ADDON_HOSTNAME", "PARLORWAVE_ADDON_IP"}, exported)

    def test_no_escape_hatches(self):
        for forbidden in ("MUSICBRAIN_AUTH_DISABLED", "MUSICBRAIN_DEBUG_DOCS",
                          "MUSICBRAIN_MUSIC_ROOT", "DJ_KOKORO_ENABLED"):
            self.assertNotIn(forbidden, self.text, forbidden)

    def test_it_hands_over_to_the_bootstrap(self):
        self.assertIn("exec python3 -m addon_bootstrap", self.text)


if __name__ == "__main__":
    unittest.main()
