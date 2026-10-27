# ParlorWave Radio (Home Assistant add-on)

The ParlorWave Radio backend for Home Assistant OS. It runs beside the
Music Assistant add-on, speaks through your Home Assistant text-to-speech
voices, and needs no GPU, no language model and no extra host.

> **Alpha.** This works end to end, and some of it will still change.
> Bug reports are wanted:
> <https://github.com/parlorwave/parlorwave-radio-addon/issues>

**Music Assistant is required** — it owns the music, the queue and the
players; ParlorWave Radio puts a station around what it is already
playing.

Setup, in order. Each step works whether or not the later ones are done
yet, so nothing here can leave you stuck:

1. **Start the add-on.** On first start it creates its installation
   identity and its API key under the add-on's own data directory, and
   announces itself to Home Assistant. It keeps running whether or not
   Music Assistant is configured yet, and says what is missing.
2. **Confirm the integration.** Settings > Devices & services shows
   **ParlorWave Radio** as discovered: confirm it. No host, port or key to
   type.
3. **Create a Music Assistant access token**: in Music Assistant, Profile >
   Access tokens. A Home Assistant token will not work.
4. **Paste the token** into this add-on's **Configuration** tab, then
   restart the add-on. The Music Assistant URL there already points at the
   Music Assistant add-on on this machine; change it only if your Music
   Assistant is somewhere else. The integration reconnects on its own --
   nothing to re-confirm, and no Home Assistant restart.
5. **Pick the DJ-enabled outputs** in the panel, under Settings > Speakers
   ParlorWave may DJ on: the Music Assistant speakers ParlorWave Radio may
   adopt playback on and DJ.
6. **Switch the DJ on.** It speaks with one of your Home Assistant voices
   straight away; choose another in the panel's **Booth** whenever you like.

At every point the add-on log and the integration's **DJ status** sensor
(its `problems` attribute) name what is still missing and what to do about
it. Steps 3 and 4 may be done before step 2 if you prefer.

The whole first-run walkthrough, including the integration, is at
<https://github.com/parlorwave/parlorwave-radio#install>.

## Licence

The ParlorWave Radio source is Apache-2.0 (`LICENSE` and `NOTICE`, beside
this file and inside the add-on's container at
`/usr/share/doc/parlorwave-radio/`).

The container itself also contains a Debian system and FFmpeg, and FFmpeg as
Debian builds it is GPL-2.0-or-later — so the image is not Apache-2.0-only
and makes no such claim. `THIRD_PARTY_NOTICES.md` names every component, its
exact version, and exact URLs for its corresponding source. It travels inside
the container too, because somebody who pulls an image has not necessarily
read a repository.
