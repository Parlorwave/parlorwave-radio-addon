# ParlorWave Radio — Home Assistant add-on repository

This is the add-on repository for **[ParlorWave
Radio](https://github.com/parlorwave/parlorwave-radio)**, a radio station
built on the music you already play in Music Assistant.

> **Alpha — `0.1.0-alpha.4`.** The second public Alpha of ParlorWave Radio. It works
> end to end; some of it will still change. Bug reports are wanted:
> [open an issue](https://github.com/parlorwave/parlorwave-radio-addon/issues).

## Add this repository

Home Assistant → **Settings → Apps → App store → ⋮ → Repositories**,
and add:

```
https://github.com/parlorwave/parlorwave-radio-addon
```

**ParlorWave Radio** then appears in the store.

Home Assistant 2026.10 renamed add-ons to *apps* and the Add-on Store to the
*App store*. On an older Home Assistant the same path is **Settings → Add-ons →
Add-on Store → ⋮ → Repositories**.

## Why there is an add-on at all

The Home Assistant integration is the *face* of ParlorWave Radio — the
entities, the services and the sidebar panel. This add-on is the *service*
behind it, and it exists because the work it does cannot be done inside Home
Assistant:

- it runs a **broadcast clock** that keeps its own time between songs, and
  decides when the DJ speaks;
- it **writes** each break deterministically and **renders** it to audio;
- it **processes audio** with FFmpeg — normalising and joining the pieces of
  a break, and proving it is finite before anything is ever queued;
- it **serves that audio** to Music Assistant over the internal add-on
  network;
- it **keeps state** — your stations, settings, voices, saved voice
  recordings and break history — in its own data directory, which survives
  a Home Assistant restart and is included in the add-on's backups.

Separating the two also means restarting Home Assistant does not restart the
station, and the add-on can be updated without touching your Home Assistant
configuration.

## What you need

| | |
|---|---|
| **Home Assistant OS** (or Supervised) | the add-on system |
| **[Music Assistant](https://music-assistant.io/)** | **required** — it owns the music, the queue and the players |
| **A Home Assistant text-to-speech voice** | Piper keeps the words local; an online voice (Home Assistant Cloud, Google Translate and others) sends each break's words to its provider |

No GPU, no language model, no local music catalogue, no account.

## The Music Assistant token is required

This is the one thing you must type. In **Music Assistant → Profile →
Access tokens**, create a token (a Music Assistant token, not a Home
Assistant one), and paste it into this add-on's **Configuration** tab.
Restart the add-on.

The `music_assistant_url` option defaults to `http://homeassistant:8095` —
the Music Assistant add-on on this same machine. Change it only if your Music
Assistant runs somewhere else.

Without the token the add-on still starts and stays up; it reports what is
missing instead of failing. Everything else — the connection between the
add-on and Home Assistant, the API key, the internal hostname — is arranged
automatically through Supervisor discovery, and you never see it.

## What the add-on can reach

Deliberately very little:

- **`homeassistant_api`** — to render speech through your Home Assistant
  text-to-speech voices, and to publish its own state. The voice you choose
  decides where the words go: a local voice keeps them on your hardware, an
  online one sends them to its provider.
- **`discovery`** — to tell Home Assistant where it is, so the integration
  appears ready to confirm.
- **No published host port.** Home Assistant and Music Assistant reach it on
  the internal add-on network. Port 5005 can be mapped by hand for debugging;
  it is unmapped by default.
- **No host filesystem access**, no host network, no privileged mode.

## Supported architectures

| Architecture | Typical hardware |
|---|---|
| `amd64` | Intel/AMD machines, NUCs, most VMs |
| `aarch64` | Raspberry Pi 4/5 (64-bit), Home Assistant Green/Yellow, ARM servers |

32-bit ARM (`armv7`, `armhf`) and `i386` are not supported; Home Assistant
has deprecated them.

## How updates work

The add-on is published as a **prebuilt multi-architecture image** at
`ghcr.io/parlorwave/parlorwave-radio-addon`. Your machine downloads a
finished image for its architecture — it never compiles apt or pip packages,
which on a Raspberry Pi is the difference between a minute and an afternoon.

When a new version is released, Home Assistant offers it under **Settings →
Apps → ParlorWave Radio** (**Settings → Add-ons** before Home Assistant
2026.10), exactly like any other add-on. Your data
directory is not touched by an update.

**Keep the integration on the same version.** The add-on and the
[integration](https://github.com/parlorwave/parlorwave-radio) are released
together and share a version number. If they differ, **Settings → Repairs**
says so. Home Assistant compares them when it starts ParlorWave Radio, so
after updating only the add-on the notice appears or clears at the next
restart, or when you reload ParlorWave Radio under **Settings → Devices &
services**.

## Full install instructions

The complete first-run walkthrough — add-on, integration, token, outputs,
voice, DJ on — lives with the integration:

**<https://github.com/parlorwave/parlorwave-radio#install>**

## Licence

The ParlorWave Radio source is Apache License 2.0 — see [LICENSE](LICENSE)
and [NOTICE](NOTICE).

The published **image** is a different artefact. It also contains a Debian
base system and FFmpeg, and FFmpeg as Debian builds it is GPL-2.0-or-later,
so the image is not Apache-2.0-only and does not claim to be. Every component,
its exact version, and exact URLs for its corresponding source are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
