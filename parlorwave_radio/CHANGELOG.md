# Changelog

## 0.1.0-alpha.4

The second public Alpha: a polish and reliability release on top of
0.1.0-alpha.3. Install the integration at the same version; its changelog
lists what changed on screen. Updating keeps your data directory, settings,
stations, voices and saved recordings.

The limitations at the end of this entry are part of the release.

**The Request Line**

- **One song, played next.** `/request/song` with no speaker named plays one
  song from your Music Assistant library next on the speaker the DJ is on,
  using Music Assistant's own "play next". It never guesses between several
  DJ speakers. A request that names a speaker behaves exactly as before.
- **It refuses rather than surprise you**, inserting nothing, when:
  - the speaker is paused, or nothing is playing;
  - a DJ break or news is on the air or next;
  - Music Assistant has already lined up the next song (the same gate DJ
    breaks wait for);
  - a previous request is still waiting ahead of the song playing;
  - the queue changes between choosing the song and inserting it;
  - Music Assistant refuses the insert.
- **"Up next" is said only when it is true.** After the insert, the service
  reads the queue back. Only when the song sits directly after the one
  playing does it answer `inserted_local_next`; otherwise it answers
  `inserted_unconfirmed`.
- **Versions are part of a recording's identity.** A studio and a live
  recording of one title are both offered, and *Title (Version)* chooses one.
  An ambiguous answer says whether the artist, the version or the album is in
  doubt.

**The DJ**

- **With an external writing model, a break never repeats an old clock.** A
  time the station did not supply for this break is refused, and the
  built-in writer takes the slot. The model's memory of recent breaks no
  longer carries times or temperatures to copy. A time written in words
  ("seven fifteen") counts as a time.
- **A model draft that only lacks the due time or weather is completed**
  with the station's own words instead of being thrown away.
- **Weather is mentioned again only when it really changes**: a new
  condition, or a temperature move of two or more steps, not a one-degree
  flicker.
- **The model's tone follows the chosen personality.**
- A closing quotation mark before a temperature is no longer read as a year,
  and the same story about an artist is not retold within twelve hours.
- **After a News bulletin, the first song's end can host a break again.**
  Before, the first boundary after News could never host, so the next break
  waited a second song.
- **A break whose speech comes back with a long silence is rendered once
  more** instead of being lost. The silence check still keeps the faulty
  take off the air.
- **Album facts come from the song's own album.** A compilation or single
  with the same title no longer supplies the year, and no year span is
  said across the two.
- **Clock times are spoken with AM or PM**, never as a number a voice
  engine reads as thousands.
- **Owed weather counts as said only when the weather was said.** A number
  or a phrase in music talk no longer pays it. "Of daylight left" and
  "just after sunset" are recorded as the daylight they describe, and a
  clear sky after sunset is "clear", not "sunny".
- **A break cut short on air is not counted as heard.** Its time and weather
  may be said again, and `/dj/status` reports the break's coverage.
- **With an external writing model, request acknowledgements and
  announcement rewrites no longer make the model reload** before the next
  break.

**Queues and playback**

- **A finished queue is reported as ended.** When a queue without Repeat
  plays to its end, the service reports it as ended, from Music Assistant's
  own evidence, confirmed twice. Stopping or pausing during the final song
  is not an ending.
- **Playing it again is a new pass.** A person's Play from Home Assistant
  after the end, Music Assistant marking the queue ended, or Repeat wrapping
  round starts a new DJ pass. Before, the DJ stayed silent over music it had
  already hosted once.
- **Spent breaks left behind at the end of a Flow queue are cleaned up** once
  Music Assistant opens the stream again, instead of being forgotten.
- **Music Assistant Autoplay refilling with old breaks is reported** as a
  warning, with the remedy. Nothing in Music Assistant is changed.
- **No program item is shorter than 5 s.** A very short announcement or
  break is padded, so a player cannot move past it before it is heard, and
  the service keeps watching an item the player has already buffered.
- **A Dial Station can take over the DJ's own speaker** by an explicit
  handover (see the integration's changelog). The songs queued ahead are
  removed only when named, and a delayed restore of Music Assistant's
  Autoplay setting never overwrites a newer choice.

**The Newsroom**

- **"Last aired" is measured in the station's time zone.** A bulletin's
  stored airing time was read as UTC, so it looked hours old everywhere
  else.
- **A source in the source list counts as configured.** A fresh install
  with a feed added under Bulletin Sources no longer reports *No feed
  configured*, or a trouble line about a missing feed URL, while bulletins
  air.

**Diagnostics and status**

- **Broadcast diagnostics.** `GET /dj/diagnostics/opportunities` and
  `GET /dj/diagnostics/opportunities/{id}`, by the break's public id, give a
  read-only account of each DJ break from what the station recorded: aired
  as written, aired as the station's own line, suppressed, or not delivered,
  and why. The API key is required. No break token, audio path or URL is
  returned, calendar titles are masked, and announcement messages are
  withheld. A break written by the built-in writer, a writing model or a
  prepared story is *accepted*.
- **`/dj/status` reports `stories_active`**: whether music stories can air.
  They cannot in this release; the listener's preference is kept.
- **The Music Assistant token messages point to Profile > Access tokens**,
  where Music Assistant keeps them: the add-on's start-up warning and the
  missing-token problem.

**Packaging**

- The `NOTICE` file now travels inside the image, beside `LICENSE` and
  `THIRD_PARTY_NOTICES.md`, in `/usr/share/doc/parlorwave-radio/`.
- **The image's contents are fixed.** It is built from a base image pinned by
  digest, takes every Debian package (the base's own included) from one
  pinned archive snapshot, and installs exactly the Python packages its
  requirements list, each at one pinned version: an image built on any day is
  the same image.
- **Newer base, security updates and certificates.** The Debian packages are
  this autumn's stable and security updates: OpenSSL 3.0.22-1~deb12u2,
  `libexpat1` 2.5.0-1+deb12u5, `libbrotli1` 1.0.9-2+deb12u1 and `tzdata`
  2026e among them, with `ca-certificates` (the service's own HTTPS requests
  use that system store). The bundled `certifi`
  is 2026.10.23, past advisory GHSA-6q48-6c7w-9gw2; the 0.1.0-alpha.3 image
  carried an affected version.

**Limitations of this Alpha**

- **Music Assistant 2.10.6, Flow mode, "smart" sample rate.** In Flow mode
  Music Assistant fixes the stream's sample rate from the first item it
  plays. A later item that needs a higher rate makes it end that stream and
  open a new one, which can be heard as a gap or a restart. Songs of
  different sample rates are enough to cause it, and ParlorWave's breaks
  (24 kHz mono MP3) can be on either side of such a change. A possible
  workaround, per speaker: in that player's settings in Music Assistant,
  change the Flow mode sample rate from *Smart* to a fixed rate such as
  *48000*, so everything is resampled instead of restarted. Which rates a
  speaker accepts depends on the player. ParlorWave Radio does not change
  this setting and does not work around it in its own audio.
- **Music Assistant Autoplay can refill a queue with old breaks.** Its
  similar-tracks refill is seeded from the last items added to the queue,
  and ParlorWave's breaks and bulletins are among them. The service keeps
  removing what it inserted, but the queue can run out of music. Remedy: set
  that player's Autoplay mode to your library or a playlist, or switch
  Autoplay off.
- **After Music Assistant restarts**, it can restore a queue at an earlier
  position and leave it stopped. Press Play.
- **The single-song request** cannot choose between albums yet, and allows
  one waiting request per speaker. It is held in memory and forgotten when
  the add-on restarts.
- **A Play from Music Assistant's own app seconds after a queue ends** can be
  taken for Music Assistant re-opening the stream rather than a new pass,
  when Music Assistant did not mark the queue ended. A Play from Home
  Assistant is always recognised.
- **Architectures.** The image is published for amd64 and aarch64 as one
  multi-architecture manifest. The aarch64 image was built, and its userspace
  and entrypoint executed, under QEMU emulation on an x86 machine. It has not
  been tested on a physical Raspberry Pi.
- **Updating from 0.1.0-alpha.3** passed when rehearsed with a staged copy of
  this version (Home Assistant OS 18.3, Home Assistant 2026.10.4, Music
  Assistant 2.10.8). The update from the published repository, through the
  App store, cannot be verified until this version is published.
- **No music-knowledge pack is included**, and **Pandora** sources are not
  qualified in this release.
- **No voice engine and no curated voices ship.** The DJ speaks with your
  Home Assistant voices. Custom voices need a Chatterbox-compatible voice
  engine that you run yourself; they were tested with Chatterbox TTS API
  2.1.0, and three are held at a time.
- **The Station Programmer needs Music Assistant's library** and plays on one
  speaker at a time.
- **Listening to the station in the browser is not in this release.**
- **Previous can replay a break or bulletin in full**, from Music
  Assistant's own memory of recently played audio, without asking
  ParlorWave. The replay is not necessarily recorded as a new airing.

## 0.1.0-alpha.3

The first public Alpha of ParlorWave Radio. It replaces the 0.1.0-alpha.1
development preview: everything that preview did is in this release,
together with everything since (0.1.0-alpha.2 was an internal release
candidate and was never published).

Alpha still means the product works end to end and its shape may still
change. The limitations at the end of this entry are part of the release.

**Installing and updating**

- **First install needs no Home Assistant restart.** The add-on announces
  itself to Home Assistant only once its service is answering, so the
  discovered **ParlorWave Radio** card appears on the first start instead of
  after the next Core restart.
- **The add-on reports its release** to Home Assistant, so a mismatched
  integration is shown under Repairs.
- **Each installation keeps its own identity.** A reinstalled add-on is a new
  installation with a new key; the old entry asks for its key and is never
  re-pointed at the new one. Delete it and confirm the newly discovered
  ParlorWave Radio.
- **The Music Assistant token option says what it is**: a long-lived access
  token created in Music Assistant (Profile > Access tokens), not a Home
  Assistant token. When Music Assistant refuses a token, ParlorWave Radio
  reports that it was refused.
- A public install no longer carries a permanent warning about a
  music-knowledge pack it was never given. Absence of the pack is a
  capability fact, not a fault (see Limitations).
- A rebuild or restart during a DJ render no longer stalls the DJ until a
  timeout; the orphaned render is cleared on the next pass.
- Every module logs through the add-on's one configured logger.

**The DJ**

- **Breaks are written by a rebuilt deterministic writer.** Every break is
  written on this machine with no language model; the optional external
  writing model, when configured, writes behind it, and the built-in writer
  takes over whenever the model's draft is rejected.
- **A due time-check is spoken exactly once, with AM or PM**, and the weather
  and what is next are honoured when they are due, on both the built-in and
  the model-written path.
- **A late clock costs one break, never a run of them.** A break written for a
  moment that has passed is rewritten once, in the same pass.
- **Station programming keeps local time across daylight-saving changes**,
  and the DJ's sense of how long ago something played stays right.
- **A prepared break re-checks its facts before it airs.** A break held back
  by a queue rewind is rewritten from what is true now, or skipped; music is
  never delayed for it.
- **The DJ can only speak on a speaker it is allowed on**, enforced in the
  service rather than hidden in a dropdown. DJ hosting can be silenced on its
  own, and a DJ voice is checked at the moment of speaking.
- **There is no voice to pick before the DJ can speak.** Until you choose
  one, the DJ speaks with your voice engine's default when one is
  configured, otherwise with one of your Home Assistant voices. A voice you
  choose becomes the DJ only after it has spoken a test phrase; one that
  cannot is refused with the reason, and the DJ stays as it was.
- **Custom voices**: Booth makes a DJ voice from a 10–30 second recording,
  with a voice engine you run yourself (see the limitations below). A
  recording is checked for length, silence and clipping before it is sent,
  and a new voice speaks a test line before it can become the DJ.
- **Replacing a voice's recording never costs it the one that works.** The
  new recording is uploaded beside the old one and must speak first; any
  failure leaves the voice exactly as it was.
- **Saved recordings.** Every recording a custom voice is made from, and
  every recording a replace or delete takes away, is kept in the add-on's
  own data directory, apart from the voice engine. Kept recordings can be
  played, renamed, used again for a custom voice (with the same permission
  step), archived, and deleted once archived and used by no voice. They are
  part of the add-on's Home Assistant backups.
- **Three more personalities**: Sarcastic, Humorous and Energetic, each a
  distinct writer profile.
- **The writing model is no longer told to skip what a break owes.** A due
  weather reading was also listed among the habits to avoid, so model drafts
  skipped it and lost the slot to the built-in writer. A break about the
  weather, the time or an event still names a song, and a break that keeps
  opening with the time is asked to put it last.
- **The station's own frequency is said the way radio says it.** "88.5" in
  the configured station name reaches the voice as "eighty-eight point five"
  (and "104.3" as "one oh four point three"); voice engines read the digits
  unreliably. Only the speech changes: what is shown, stored and logged
  keeps your spelling, and no other number is touched.
- **All-day calendar events can be mentioned**, once, on their own day or the
  day before, and the DJ never invents a clock time for one.
- **The calendar and the weather can be inspected.** Read-only routes report
  what the station can actually see and why each event or reading would or
  would not reach a break.
- What actually aired is recorded as it happened: a break is counted when it
  completes, and only when Music Assistant confirms it is the item playing.

**The Newsroom**

- **Bulletins air west of UTC.** Freshness compared a UTC timestamp with a
  local-clock one, so on any installation not running UTC every bulletin was
  rejected as published in the future.
- **A due bulletin owns its boundary; the DJ break defers to it.**
- **The News block stays together on a shuffled queue.** With shuffle on in
  Music Assistant, the bulletin and its intro still play together; a bulletin
  that can no longer be placed together is retired instead of airing late.
- **More than one bulletin source**, each with its own schedule, its own
  last-aired record and its own test. A single existing feed migrates on its
  own.
- **A record of what the Newsroom actually did**: aired, skipped or failed,
  with the time it really went out.
- **A bulletin is queued once.** ParlorWave never queues one that aired
  again, even after a restart, and never retires one while it is on the air.
  Going back to it with Previous is a different matter: see Limitations.
- **A news intro queued well ahead of its bulletin states no clock time.** An
  intro rendered more than three minutes before it airs says the same thing
  without a time, so a skipped song can never make it announce the wrong
  hour.
- Thirty minutes is the most a bulletin may be; longer episodes are refused
  with the reason.

**The Station**

- **Announcements.**
  - A calendar event whose summary begins with the word `ANNOUNCE` is station
    programming. It airs once per occurrence, at a music boundary at its
    scheduled time.
  - It is spoken verbatim after `ANNOUNCE EXACT`, or with its times read
    naturally otherwise.
  - An event can name an authorized saved voice (`voice: <name>` in its
    description).
  - It works with or without a DJ and with or without a writing model.
- **The Station Programmer.**
  - Name a station ("90s Rock", an artist, a decade) and ParlorWave programs
    it from Music Assistant's own library.
  - Test Station is a real dry run of the production builder, and Dial
    presets hold the stations you keep.
  - Start Station plays one on a single speaker you name, with an endless
    runway and a Stop. Your selected DJ speaker is refused unless you authorize
    it explicitly.
  - A switch between stations keeps the song a queued DJ break has already
    promised.
- **The Request Line** starts a station from a typed request through the same
  builder, and the console's dial starts a tested preset.
- **The on-air identity is configurable**: the station name the DJ uses.
- **Queue hygiene, and recovery after a long interruption.**
  - After a queue rewind, a long pause, an outage, or a re-attach to a queue,
    ParlorWave removes its own spent breaks and bulletins from ahead of the
    play head, in one pass, so a rewind never meets a run of silences.
  - It removes only items it inserted itself, by exact id. It never removes
    music, and never the item Music Assistant has named as next.
  - A cleanup it could not do yet is kept owed, never dropped. A rewind that
    spans a Music Assistant outage is reconciled as one rewind.
- **Speakers outside Flow mode** no longer stop when a retired item is skipped
  between two songs.
- A station start whose proof cannot be read for fifteen minutes is given up
  rather than blocking every later start.
- **The station stops rewriting what it already wrote**: unchanged state is
  not written back to storage every cycle, for the sake of SD cards.

**Limitations of this Alpha**

- **Architectures.** The image is published for amd64 and aarch64 as one
  multi-architecture manifest. The aarch64 image was built, and its userspace
  and entrypoint executed, under QEMU emulation on an x86 machine. It has not
  been tested on a physical Raspberry Pi.
- **Updating from the 0.1.0-alpha.1 preview through the add-on store has not
  been rehearsed**, because it needs a published version to update to.
- **No music-knowledge pack is included.** The DJ has no library of artist
  and album facts to draw on in this release.
- **Pandora** sources are not qualified in this release.
- **No voice engine and no curated voices ship.** The DJ speaks with your
  Home Assistant voices. Custom voices need a Chatterbox-compatible voice
  engine that you run yourself and add under Settings > Optional services;
  without one, Booth shows "Add voice" switched off with the reason. They
  were tested with Chatterbox TTS API 2.1.0, and three are held at a time.
  Recording in the browser needs Home Assistant opened over https; otherwise
  choose a WAV, MP3 or FLAC file. A voice uploaded with the 0.1.0-alpha.1
  preview was stored before the permission step existed; it keeps working,
  and Booth says "Permission not recorded" beside it.
- **The Station Programmer needs Music Assistant's library** and plays on one
  speaker at a time.
- **Listening to the station in the browser is not in this release.**
- **Previous can replay a break or bulletin in full.** When Music Assistant
  asks for a break or bulletin whose time has passed, ParlorWave answers with
  a few seconds of silence. Music Assistant keeps recently played audio in
  memory, though, and when Previous or a rewind goes back to something it
  played recently, it can play its own copy of the original again without
  asking ParlorWave. You hear the break or bulletin again, and the replay is
  not necessarily recorded as a new airing in Break History. A rewind that
  Music Assistant makes on its own could replay a programme the same way;
  that is a known limitation, and it has not been observed.

## 0.1.0-alpha.1 (development preview)

First public Alpha.

- The ParlorWave Radio backend as a Home Assistant add-on, published as a
  prebuilt multi-architecture image (amd64, aarch64) so nothing is compiled
  on your machine.
- Reaches Home Assistant and Music Assistant on the internal add-on network:
  no host port is published and no API key is typed anywhere.
- Announces its connection details to Home Assistant through Supervisor
  discovery, so the **ParlorWave Radio** integration appears ready to
  confirm.
- Speaks through your existing Home Assistant text-to-speech voices, via the
  Core API proxy.
- Starts and stays up before Music Assistant is configured, and reports what
  is still missing rather than failing.
- Contains no copyleft Python library: audio durations are read through the
  `ffprobe` executable. The image itself contains Debian and a GPL-enabled
  FFmpeg, and carries no image-wide licence claim; `THIRD_PARTY_NOTICES.md`
  travels inside it with the exact versions and exact source URLs.

Alpha means the product works end to end and its shape may still change.
