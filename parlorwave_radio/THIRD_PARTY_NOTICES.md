# Third-party notices

Three different things have three different licence positions, and this file
keeps them apart on purpose. Conflating them is how a project ends up making
a claim about software it did not write.

| | What it is | Licence position |
|---|---|---|
| **1. ParlorWave source** | every file in this repository | Apache-2.0 (`LICENSE`) |
| **2. Third-party runtime packages** | Python distributions the image installs from PyPI at build time | each keeps its own licence; listed below |
| **3. The container image** | 1 + 2 + a Debian system and FFmpeg | an aggregate of many licences, several copyleft; **not** Apache-2.0-only |

Nothing in category 2 or 3 is redistributed *in this repository*. All of it is
present *in the published image*, which is why the notices exist.

---

## 1. ParlorWave source

Apache License 2.0. Every file here is original ParlorWave work: no
third-party source code is vendored, copied or embedded, and no GPL library
is imported by anything that ships. The one third-party work inside a
ParlorWave file is typographic, not code: the Home Assistant panel embeds a
subset of the Jost typeface for its badge and draws its wordmark from
outlines derived from Mr Dafoe, both under the SIL Open Font License 1.1
(section 7). One image in that file is generated, not drawn: the fallback
news illustration was made with ChatGPT for ParlorWave Radio. It is not
third-party work, and `NOTICE` says what is and is not claimed for it.

That last point was work rather than luck. Duration probing and ID3 tagging
used to be done with `mutagen`, which is GPL-2.0-or-later. Importing it made
the running service a combined work with a copyleft library and put GPL
software inside every image. It is gone: durations and stream facts are read
through the `ffprobe` executable, and the handful of ID3 text frames the
product writes are written directly, in `audio_probe.py`. `mutagen` is not a
dependency, is not installed in the image, and is not importable there.

---

## 2. Python packages installed into the image

Every package below is pinned to this exact version by
`parlorwave_radio/backend/requirements.txt` -- the transitive ones included --
and installed with `--no-deps`, so nothing else can arrive. Checked against
clean, no-cache builds of `0.1.0-alpha.4` on both architectures; `pip` itself
is Debian's `python3-pip`:

| Package | Version | Licence |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| cffi | 2.1.1 | MIT-0 |
| click | 8.5.0 | BSD-3-Clause |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| fastapi | 0.139.0 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| pip | 23.0.1 | MIT |
| pycparser | 3.12 | BSD-3-Clause |
| pydantic | 2.13.4 | MIT |
| pydantic_core | 2.46.4 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.4 | Apache-2.0 |
| uvicorn | 0.50.0 | BSD-3-Clause |

All permissive; `certifi` alone is weak copyleft (MPL-2.0, per file) and is
installed unmodified from PyPI. Each package's own licence text is inside the
image under its `*.dist-info/licenses/` directory, and the list above can be
regenerated from any running container.

---

## 3. FFmpeg

**FFmpeg is GPL-2.0-or-later in this image**, not LGPL. Debian builds it with
`--enable-gpl`, and Debian's own copyright file — which ships in the image at
`/usr/share/doc/ffmpeg/copyright` — states it plainly:

> For building the default Debian packages some of the GPL licensed files are
> used, so the resulting binaries are licensed under GPL v2+.

ParlorWave Radio invokes `ffmpeg` and `ffprobe` as **separate programs**, with
command-line arguments, to transcode speech, analyse breaks for silence,
generate matching silence frames and read durations. It does not link against
`libav*`, embed FFmpeg code, or ship a modified FFmpeg.

### Exactly what is in the image

Verified inside both published architectures; they are the same package and
the same version.

| | |
|---|---|
| Binary package | `ffmpeg` |
| Version | `7:5.1.9-0+deb12u1` (upstream 5.1.9; the `7:` is Debian's epoch) |
| Source package | `ffmpeg` `7:5.1.9-0+deb12u1` |
| Architectures | `amd64`, `arm64` — identical version on both |
| Distribution | Debian 12.15 (`bookworm`) |
| Also from this source | `libavcodec59`, `libavdevice59`, `libavfilter8`, `libavformat59`, `libavutil57`, `libpostproc56`, `libswresample4`, `libswscale6`, all at the same version |

### How to obtain the corresponding source

The image installs every Debian package, the base image's own included,
from one pinned archive snapshot, so these are exact
URLs rather than a promise. Download the three components of the source
package and unpack them with `dpkg-source -x`:

```
http://snapshot.debian.org/archive/debian/20261028T143257Z/pool/main/f/ffmpeg/ffmpeg_5.1.9-0+deb12u1.dsc
http://snapshot.debian.org/archive/debian/20261028T143257Z/pool/main/f/ffmpeg/ffmpeg_5.1.9.orig.tar.xz
http://snapshot.debian.org/archive/debian/20261028T143257Z/pool/main/f/ffmpeg/ffmpeg_5.1.9-0+deb12u1.debian.tar.xz
```

SHA-256, as recorded in the `.dsc`:

```
d9b593bb2ba93d4b50f74177e0cdcd41747e708596367deed0c30348a71dd176  ffmpeg_5.1.9.orig.tar.xz
71b20472ecd2764bc98e3229d34c0f7c8dcfe4a6cbacc23fd5604c23bac04632  ffmpeg_5.1.9-0+deb12u1.debian.tar.xz
```

Or browse it: <https://sources.debian.org/src/ffmpeg/7%3A5.1.9-0%2Bdeb12u1/>

Unmodified upstream releases are at <https://ffmpeg.org/download.html>, and
`snapshot.debian.org` keeps the Debian packaging permanently, so neither
location depends on ParlorWave continuing to exist.

---

## 4. The Debian base system

The image is built `FROM ghcr.io/home-assistant/base-debian:bookworm`, pinned
by digest (`sha256:ebb912a11616e3eda2d31204803587276e70bbb4e7f02a6c92dba5f43256b48d`,
base-debian 2026.10.3), Home Assistant's Debian 12 base. ParlorWave does not
build that base; what follows was read out of the finished image.

### The init system and the add-on helper

Alongside Debian, the base contributes the s6 process supervisor and Home
Assistant's shell helper. Exact versions, from `/package/admin` in the image:

| Component | Version | Licence |
|---|---|---|
| `s6-overlay` | 3.2.3.0 | ISC |
| `s6` | 2.15.0.0 | ISC |
| `s6-rc` | 0.6.1.0 | ISC |
| `s6-linux-init` | 1.2.0.1 | ISC |
| `s6-linux-utils` | 2.6.4.1 | ISC |
| `s6-portable-utils` | 2.3.1.2 | ISC |
| `s6-overlay-helpers` | 0.1.2.2 | ISC |
| `execline` | 2.9.9.0 | ISC |
| `bashio` | (`/usr/bin/bashio`) | MIT |

All permissive. **Their licence texts are not present in the image** — the
base image does not ship them, and ParlorWave has no way to add them to a
layer it did not build. They are at
<https://github.com/just-containers/s6-overlay> (`COPYING`),
<https://git.skarnet.org/cgi-bin/cgit.cgi> for the skarnet components, and
<https://github.com/hassio-addons/bashio> (`LICENSE.md`).

### Debian itself

The image contains **323 Debian packages on amd64 and 319 on arm64**, of which 176
carry a GPL, LGPL or MPL licence line — predominantly GPL-2+, GPL-3+ and
LGPL-2.1+, alongside MIT, BSD and public domain. Every one of them ships its
own licence text inside the image.

The exact inventory is a property of the image and is readable from it, which
is more reliable than a list copied into a document:

```sh
# every package and version
docker run --rm ghcr.io/parlorwave/parlorwave-radio-addon:<version> \
  dpkg-query -W -f='${Package} ${Version} ${source:Package} ${source:Version}\n'

# one package's full licence text
docker run --rm ghcr.io/parlorwave/parlorwave-radio-addon:<version> \
  cat /usr/share/doc/<package>/copyright
```

### How to obtain the corresponding source for any of them

The image's `/etc/apt/sources.list.d/debian.sources` records the exact
snapshot it was built from:

```
http://snapshot.debian.org/archive/debian/20261028T143257Z          (bookworm, bookworm-updates)
http://snapshot.debian.org/archive/debian-security/20261028T143257Z  (bookworm-security)
```

Point `apt` at that snapshot and fetch the source for any package in the
image, at exactly the version the image has:

```sh
printf 'deb-src http://snapshot.debian.org/archive/debian/20261028T143257Z bookworm main\n' \
  > /etc/apt/sources.list.d/snapshot-src.list
apt-get -o Acquire::Check-Valid-Until=false update
apt-get source <package>
```

Sources are also browsable at <https://sources.debian.org/>.

---

## 5. Image licence metadata

The published image carries **no `org.opencontainers.image.licenses`
annotation**, and that is deliberate.

That annotation describes the licence of the software *contained in the
image*. This image contains a Debian system, a GPL-enabled FFmpeg and several
hundred packages under a mixture of licences, so `Apache-2.0` there would be a
false statement about nearly all of it — and no single SPDX expression
truthfully describes the aggregate without someone inventing one that has not
been verified. The annotation is optional, so it is omitted rather than
guessed at.

What is true is labelled instead, scoped so it cannot be read as a claim about
the whole image:

| Label | Value |
|---|---|
| `io.parlorwave.source.license` | `Apache-2.0` — ParlorWave's own source, and only that |
| `io.parlorwave.third-party-notices` | `/usr/share/doc/parlorwave-radio/THIRD_PARTY_NOTICES.md` |
| `org.opencontainers.image.source` | the add-on repository |

This file and the Apache licence text are inside the image at that path,
because somebody who pulls an image has not necessarily seen the repository,
and the copyleft components are precisely the ones whose source instructions
have to travel with the binary.

The `LICENSE` file in this repository is about this repository's source, and
is accurate on its own terms regardless of anything above.

---

## 6. What is deliberately absent

The first public Alpha ships no local voice engine and no model runtime.
Verified by inspection of both built images: **no** PyTorch, ONNX Runtime,
NumPy, Kokoro, Chatterbox or espeak-ng, and no model weights anywhere on the
filesystem — and therefore none of the GPL-3.0 and mixed-licence native
material such an engine would bring with it. ParlorWave Radio speaks through
Home Assistant's own text-to-speech instead.

---

*Nothing in this file is legal advice. It records what was measured inside the
published images and where the corresponding sources are.*

---

## 7. Typefaces in the Home Assistant panel

The On Air console's badge (`custom_components/parlorwave_radio/panel/parlorwave-radio-panel.js`)
carries two pieces of third-party typography. Neither is a Python package and
neither is in the image's Debian or FFmpeg layers; they travel inside the one
JavaScript file the integration serves.

| Typeface | Copyright | Licence | How it is used |
|---|---|---|---|
| Jost | Copyright 2020 The Jost Project Authors (https://github.com/indestructible-type); no Reserved Font Name | SIL Open Font License 1.1 | A Latin subset at weight 600 (about 9 KB of WOFF2) is embedded base64-encoded as `PW_PLATE_FONT_B64` and registered at run time under the CSS family name "ParlorWave Plate" for the engraved station-name plate. The subset keeps the font's copyright and licence records, and the full OFL text is reproduced in the panel beside the embedded data, as the licence requires of every copy of the Font Software. The family is not renamed. |
| Mr Dafoe | Copyright 2011 Alejandro Paul (Sudtipos), Reserved Font Name "Mr Dafoe" | SIL Open Font License 1.1 | The ParlorWave wordmark (`PW_EMBLEM_SVG`) is vector artwork whose lettering was outlined, re-spaced and joined from this typeface. No Mr Dafoe font file is embedded or shipped; the OFL treats artwork made with a font as outside the licence, and no derived font exists. |

The OFL permits use, embedding, modification and redistribution of the Font
Software, commercially or not; its conditions are that the Font Software is
not sold by itself, that every copy carries the licence, that a modified
font does not use a Reserved Font Name, and that the authors' names are not
used to promote a derived work. The panel meets all four: the Jost subset is
not sold and carries its licence; Jost declares no Reserved Font Name, so the
subset may keep its name; no Mr Dafoe font exists to rename; and neither
author is named anywhere but in these notices and the licence text.
