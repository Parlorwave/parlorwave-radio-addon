"""DJ personality profiles, and the airtime policy they are NOT allowed
to move: WHO is talking, kept apart from HOW MUCH.

Pure module. It owns the built-in personality catalog, the chattiness
levels, and the deterministic policies each one maps to. dj_engine reads the
policies and folds the delivery guidance into its ONE break prompt; nothing
here writes settings, touches a queue or calls a provider.

PERSONA AND AIRTIME ARE DECOUPLED. A personality is a VOICE:
tone, humor, energy, the angles it reaches for. It is not, and must never
again be, a cadence setting. Picking Late Night used to silently resolve
the station to LOW chattiness -- a different word budget, a shallower
callback depth and quiet-slot suppression -- so a listener who thought they
were changing the DJ's character had also, invisibly, changed how often the
DJ spoke. Picking Dry Wit silently put it back. That coupling is gone:
`effective_chattiness` now answers PRODUCTION_CHATTINESS for every
personality and for every persisted value, and no profile carries a
chattiness default to be consulted.

Personality is delivery, never fact. Every profile writes inside the same
envelope: the facts block, the timing rules, the
service-info rules and the no-invention rules are shared and unchanged; a
profile only chooses the angle, humor and energy the shared rules are
delivered with. A profile therefore cannot make the weather wrong, invent a
callback, or talk about machinery -- those rules live in dj_engine, outside
any profile's reach.

Chattiness is an EDITORIAL MODE, not a volume knob, and never makes breaks
more frequent than the cadence (the cadence is a hard floor between audible
breaks). It decides how much of a host this station wants: what optional
material a break may draw on, how long a break should aim to be, and, at
LOW, whether a purely discretionary cadence slot airs at all.

NORMAL is the single production policy. LOW and HIGH remain below as an
INTERNAL seam -- tests, offline bake-offs and research drive them by
passing a level directly to `chattiness_policy` / `word_budget`, or by
putting one in a render context -- and nothing a customer can select
reaches them. The known LOW repeated-quiet-slot defect is therefore
unreached by production and is deliberately left unfixed here.

  low     no callback material, at most one contextual topic, a short
          target, and a cadence slot with nothing worth saying (no
          calendar/news topic selected) is skipped ONCE -- the next slot
          always airs. The DJ still speaks on cadence when there is
          something to say.
  normal  callbacks optional, topics per break_length, a main angle with
          room to finish it.
  high    deeper callback material (six earlier tracks instead of four),
          a scaled ceiling and a paragraph-sized target -- a real host
          when the material is there, and silence when it is not.

Chattiness must be able to lengthen a break. If break_length only ever
set a ceiling and nothing asked for anything else, a station set to long
and chatty would still produce single aphoristic sentences. word_budget
below is that lever; length is still earned, never padded.

Field names are the first-cut persona schema: persona_id,
display_name, description, voice_default, humor_level, energy_style,
aside_tendency, plus the guidance lines. `chattiness_default` was removed
when persona and airtime were decoupled: a stylistic profile has no opinion
about airtime, so there is no field left for production to consult.
Nothing beyond what persona selection needs is modelled; sliders and per-trait points are
deliberately absent.
"""

from __future__ import annotations

from typing import Any

CHATTINESS_LEVELS = ("low", "normal", "high")
DEFAULT_CHATTINESS = "normal"

#: The one level normal production ever runs at. Every persona resolves to
#: it, and so does every legacy persisted `dj.persona.chattiness` value
#: (null, "low", "normal", "high"). A future "DJ Frequency" control is
#: expected to change SCHEDULING, not to reopen this.
PRODUCTION_CHATTINESS = DEFAULT_CHATTINESS

# Deterministic policy per level. callback_depth is how many of the most
# recently played tracks the prompt may offer as callback material (0 =
# the earlier_this_session fact is never supplied, so no callback can be
# grounded); max_topics_cap bounds the contextual topic budget on top of the
# break_length policy; quiet_slots enables the once-per-slot skip.
#
# words_factor and target_share are the LENGTH levers. Without them
# chattiness has no length lever at all, and the
# result on a station set to long+chatty is a stream of single aphoristic
# sentences. Nothing was
# truncating them: nothing had ever ASKED for length. break_length set a
# ceiling, the prompt quoted only that ceiling, and a model told "at most
# 90 words" is perfectly obedient at eight.
#
#   words_factor  scales the break_length ceiling, so chattiness is a
#                 material difference rather than the same break with a
#                 different footnote.
#   target_share  the editorial TARGET as a fraction of that scaled
#                 ceiling: (low, high). A target is what the writer aims
#                 for when the material is there -- never a minimum, and
#                 never a reason to pad. Weak material still earns one
#                 short sentence or silence.
CHATTINESS_POLICY: dict[str, dict[str, Any]] = {
    "low":    {"callback_depth": 0, "max_topics_cap": 1, "quiet_slots": True,
               "words_factor": 0.7, "target_share": (0.25, 0.50)},
    "normal": {"callback_depth": 4, "max_topics_cap": None,
               "quiet_slots": False, "words_factor": 1.0,
               "target_share": (0.40, 0.70)},
    "high":   {"callback_depth": 6, "max_topics_cap": None,
               "quiet_slots": False, "words_factor": 1.4,
               "target_share": (0.45, 1.0)},
}

#: Roughly how many spoken words land in a second of radio (~150 wpm).
#: Only used to document the bands below in human terms; nothing computes
#: a duration, because nothing can know one before the synthesizer runs.
WORDS_PER_SECOND = 2.5


def word_budget(base_max: int, chattiness: str | None) -> dict[str, int]:
    """The word ceiling and the editorial target for one break.

    `base_max` is the break_length ceiling (35 / 60 / 90). Returns
    {max_words, target_low, target_high}. At the default chattiness the
    ceiling is unchanged, so an installation that never touched the
    setting sees exactly what it always saw.

    Indicative bands at break_length=long (base 90), against the
    product targets they were tuned to:
        quiet   ceiling  63   target 16-32 words   (~6-13 s;  want 5-12)
        normal  ceiling  90   target 36-63 words   (~14-25 s; want 10-25)
        chatty  ceiling 126   target 57-126 words  (~23-50 s; want 20-45,
                                                    rare ~60)
    """
    policy = chattiness_policy(chattiness)
    base = max(1, int(base_max or 1))
    ceiling = max(1, int(round(base * float(policy["words_factor"]))))
    lo_share, hi_share = policy["target_share"]
    low = max(1, int(round(ceiling * float(lo_share))))
    high = max(low, min(ceiling, int(round(ceiling * float(hi_share)))))
    return {"max_words": ceiling, "target_low": low, "target_high": high}


_CHATTINESS_GUIDANCE = {
    "low": ("Talk budget: one thought, tightly delivered. No asides, no "
            "callbacks, no second angle -- say the thing and hand back to "
            "the music."),
    "normal": ("Talk budget: a main angle and room to finish it -- "
               "commonly one to three sentences. A light aside is welcome "
               "when it earns its place, never as filler."),
    "high": ("Talk budget: this listener wants a host, not a station "
             "ident. When the material is genuinely there, take the room "
             "-- a real paragraph, commonly three to six sentences: open "
             "on something the listener can hear you mean, develop the "
             "thought instead of abandoning it after one line, let your "
             "personality show, and hand off if a handoff is natural. "
             "Linger on what a track or a transition is doing; connect "
             "the song that ended to the one coming; pick up something "
             "you said earlier; fold in the weather or their calendar "
             "where it belongs. But length is earned exactly like airtime "
             "is: if there is only one good sentence in this moment, say "
             "that sentence and stop, and if there is nothing, say "
             "nothing. Never pad to fill the room."),
}

# Words the on-air copy must never use for the station's own workings. The
# shared rule below is emitted for every profile; the same list is what the
# tests scan every profile's own guidance text for.
MACHINERY_TERMS = ("arc", "stage", "queue", "session", "algorithm",
                   "playlist engine", "system", "data", "model", "prompt")

PROFILES: dict[str, dict[str, Any]] = {
    "dry_wit": {
        "persona_id": "dry_wit",
        "tone_home": "dry",
        "tone_map": {},
        "display_name": "Dry Wit",
        "description": ("Deadpan and dry with a little smugness, sincere "
                        "about the music. The house voice."),
        "voice_default": None,
        "humor_level": "dry",
        "energy_style": "unhurried",
        "aside_tendency": "occasional",
        "delivery": (
            "Deadpan and dry: understatement over emphasis, a raised "
            "eyebrow rather than a laugh track. A little smug is fine; "
            "frantic, corny or desperate never is.",
            "Sincere about the music even when wry about everything else: "
            "the joke is never at a song's expense.",
            "No exclamation points, no hype words, no 'folks'.",
        ),
        "angles": ("dry humor", "wordplay on an artist or song name",
                   "a dry observation about a song's mood or energy, drawn only "
                   "from its listed character line",
                   "a self-aware radio-DJ aside", "a light quip grounded in "
                   "any extra context listed below"),
        "ack": "Deadpan and dry; a nod, not a cheer.",
    },
    "classic_fm": {
        "persona_id": "classic_fm",
        "tone_home": "warm",
        "tone_map": {'dry': 'warm'},
        "display_name": "Classic FM",
        "description": ("Warm, bright, broadcast-polished. Music first, "
                        "one clean line, no sarcasm."),
        "voice_default": None,
        "humor_level": "light",
        "energy_style": "bright",
        "aside_tendency": "rare",
        "delivery": (
            "Warm and bright, polished like a real FM station: friendly, "
            "confident, music first. Enthusiasm is genuine, never shouted.",
            "No sarcasm, no smugness, no jokes at anyone's expense. Humor "
            "is light and quick -- one clean line at most.",
            "When next_track is listed, hand off to it cleanly by name.",
        ),
        "angles": ("a clean handoff between the two songs",
                   "an appreciative note on the music's feel",
                   "a light, friendly quip", "a quick contextual remark "
                   "grounded in any extra context listed below"),
        "ack": "Warm and bright, like a real FM handoff.",
    },
    "late_night": {
        "persona_id": "late_night",
        "tone_home": "understated",
        "tone_map": {'energized': 'understated'},
        "display_name": "Late Night",
        # "Speaks less" was removed from this description when persona and
        # airtime were decoupled: it described a cadence this profile no
        # longer sets. Short sentences are a STYLE and still true.
        "description": ("Low-key, intimate and unhurried, wry rather than "
                        "punchy. Short sentences, room to breathe."),
        "voice_default": None,
        "humor_level": "wry",
        "energy_style": "low",
        "aside_tendency": "reflective",
        "delivery": (
            "Low-key and intimate, like the closing hours of a shift: "
            "short sentences, an unhurried pace, room to breathe.",
            "Wry, not punchy. Prefer texture and mood language to jokes; "
            "no exclamations, no hype, no punchlines that need a drum hit.",
            "Never pretend it is later or earlier than the listed "
            "station_time or daylight say it is.",
        ),
        "angles": ("a reflective note on the music's mood or texture",
                   "a quiet reflection on the current song's listed character", "a wry aside", "a soft remark grounded in "
                   "any extra context listed below"),
        "ack": "Low-key and unhurried; a quiet nod.",
    },
    # Three more profiles. Each is a distinct
    # delivery profile, not a relabelled dry_wit: the guidance below is what
    # actually reaches the writer, and test_profiles_do_not_converge proves
    # no two profiles render the same block.
    "sarcastic": {
        "persona_id": "sarcastic",
        "tone_home": "dry",
        "tone_map": {},
        "display_name": "Sarcastic",
        "description": ("Sharper than Dry Wit and enjoys it -- barbed, "
                        "playful, never mean."),
        "voice_default": None,
        "humor_level": "barbed",
        "energy_style": "unhurried",
        "aside_tendency": "frequent",
        "delivery": (
            "Sarcastic and sharp-tongued: say the opposite of the obvious "
            "thing and let the gap do the work. Where Dry Wit raises an "
            "eyebrow, you raise it higher and hold it a beat longer.",
            "Playful, never cruel. The target is a situation, a cliche, a "
            "genre's own habits, or yourself -- and the barb stays light "
            "enough that the listener is in on it rather than under it.",
            "Never mock the listener, never belittle an artist or their "
            "work, and never aim downward at anyone. Sincerity about the "
            "music survives the sarcasm: you can be rude about a decade and "
            "still respect the song.",
            "One barb per break. Sustained sneering is not a personality.",
        ),
        "angles": ("a barbed remark about a musical cliche the song walks "
                   "straight into",
                   "mock-solemn overstatement about something trivial",
                   "a dry needle aimed at yourself or at radio itself",
                   "an arch observation drawn only from the current song's "
                   "listed character line",
                   "a pointed quip grounded in any extra context listed below"),
        "ack": "Sarcastic and arch; the nod comes with a raised eyebrow.",
    },
    "humorous": {
        "persona_id": "humorous",
        "tone_home": "casual",
        "tone_map": {'dry': 'casual'},
        "display_name": "Humorous",
        "description": ("Light, funny radio company -- playful observations "
                        "rather than a routine."),
        "voice_default": None,
        "humor_level": "playful",
        "energy_style": "easy",
        "aside_tendency": "occasional",
        "delivery": (
            "Light and funny in the way good radio company is funny: "
            "playful noticing, an amused turn of phrase, the small absurdity "
            "in an ordinary moment.",
            "You are not doing a routine. No setup-then-punchline shape, no "
            "callback tags, no 'what is the deal with', no laughing at your "
            "own line. If a break reads like material, rewrite it as a "
            "remark.",
            "Not every break needs a joke, and a break with nothing funny in "
            "it is a good break. Warmth carries the ones that are simply "
            "pleasant; reaching for a laugh that is not there is worse than "
            "letting the song go.",
            "The humor is never at a listener's or an artist's expense.",
        ),
        "angles": ("an amused observation about something ordinary",
                   "a playful turn of phrase on an artist or song name",
                   "gentle comedy in the contrast between two songs",
                   "a light remark on the current song's listed character",
                   "a good-humoured aside grounded in any extra context "
                   "listed below"),
        "ack": "Light and playful; a grin rather than a gag.",
    },
    "energetic": {
        "persona_id": "energetic",
        "tone_home": "energized",
        "tone_map": {'dry': 'energized', 'understated': 'casual'},
        "display_name": "Energetic",
        "description": ("Upbeat and punchy, with forward momentum. Built "
                        "for mornings and high-energy hours."),
        "voice_default": None,
        "humor_level": "light",
        "energy_style": "high",
        "aside_tendency": "rare",
        "delivery": (
            "Upbeat and punchy: short sentences, strong verbs, forward "
            "momentum. Every line should feel like it is moving toward the "
            "music rather than delaying it.",
            "Energy comes from pace and word choice, not volume. At most "
            "one exclamation point in a break and often none; no all-caps, "
            "no shouting, no 'HUGE', no stacked superlatives, no hype words "
            "repeated for emphasis.",
            "Confident and positive without becoming a commercial: you are "
            "glad about the music, not selling it.",
            "Land the handoff hard. The last line is a push into the next "
            "song, not a trailing thought.",
        ),
        "angles": ("a punchy push into the next song",
                   "an energised note on the music's drive or tempo",
                   "a quick bright remark on the current song's listed "
                   "character",
                   "a fast, upbeat quip",
                   "a brisk contextual remark grounded in any extra context "
                   "listed below"),
        "ack": "Upbeat and punchy; a push rather than a nod.",
    },
}

DEFAULT_PROFILE = "dry_wit"

#: Pre-release ids that older settings documents may still carry.
#: Read and normalized on load, persisted canonical at the next write;
#: never exposed by the catalog, never accepted as a new public id.
LEGACY_PROFILE_IDS = {"dry_jarvis": "dry_wit"}


def tone_for(profile: dict[str, Any], occasion_tone: str | None,
             occasion: bool = True) -> str:
    """The Smart DJ's editorial "Tone:" as THIS host would take it.

    The occasion (a contrast, an album run, a calendar nod) proposes a
    register; the host decides how it sounds. With no occasion the host's
    own register is used (`tone_home`) -- telling every host "dry"
    then would give an Energetic or Classic FM host a deadpan cue
    their own delivery block contradicts. A proposed register the persona
    would never use is moved to the nearest one it does (`tone_map`).
    Dry Wit, the house voice, keeps every occasion tone exactly as it was."""
    home = profile.get("tone_home") or "dry"
    if not occasion or not occasion_tone:
        return home
    return (profile.get("tone_map") or {}).get(occasion_tone, occasion_tone)


def canonical_profile_id(persona_id: Any) -> Any:
    """The public id for a stored/incoming profile id (legacy ids mapped);
    anything else is returned unchanged for the caller to validate."""
    return LEGACY_PROFILE_IDS.get(str(persona_id or ""), persona_id)

_PALETTE = ("Describe music with a wide palette (mood, texture, tempo, "
            "swagger, shimmer, grit) -- do NOT default to 'heavy', "
            "'lighter', or 'a change of pace'.")

# The shared envelope line every profile is delivered inside. Personality
# never reaches the facts: it is emitted in the delivery block, and the
# machinery rule is part of that block so no profile can drop it.
MACHINERY_RULE = ("Never mention the station's own workings: no stages, "
                  "arcs, queues, sessions, systems, algorithms, data, models "
                  "or prompts, and never describe yourself as software.")


def profile_ids() -> tuple[str, ...]:
    return tuple(PROFILES)


def get_profile(persona_id: str | None) -> dict[str, Any]:
    """The named profile, or the default when the id is unknown/None. The
    caller has already validated stored settings; this is the render-time
    fallback so an unexpected value degrades to the house voice rather than
    to silence."""
    return PROFILES.get(str(canonical_profile_id(persona_id) or ""),
                        PROFILES[DEFAULT_PROFILE])


def effective_chattiness(dj_settings: dict[str, Any] | None = None) -> str:
    """The level in force for a NORMAL production render: always
    PRODUCTION_CHATTINESS.

    `dj_settings` is still accepted so every existing call site is valid
    and so a stored document is still a legal input, but nothing in it is
    consulted. It used to be: an explicit `dj.persona.chattiness` won, and
    null followed the selected profile's own `chattiness_default`. Both
    halves of that are deliberately gone -- the profile half because it
    made personality a hidden airtime control (Late Night resolved LOW),
    and the settings half because a value persisted under the old UI must
    not keep steering a station whose owner can no longer see or change
    it. Legacy values stay readable on disk and stay valid to the schema;
    they simply decide nothing.

    Research and tests that WANT a non-production level do not come
    through here: they call `chattiness_policy` / `word_budget` with the
    level, or put `chattiness` in the render context, which the packet
    builders still honour.
    """
    return PRODUCTION_CHATTINESS


def chattiness_policy(level: str | None) -> dict[str, Any]:
    return dict(CHATTINESS_POLICY.get(str(level or ""),
                                      CHATTINESS_POLICY[DEFAULT_CHATTINESS]))


def delivery_block(profile: dict[str, Any], chattiness: str) -> str:
    """The personality section of the break prompt: delivery guidance,
    the angle menu, the shared palette rule, the machinery rule and the
    chattiness talk budget. One block, emitted where the old fixed Style
    line lived -- never bolted onto the end after the facts."""
    angles = ", ".join(profile.get("angles") or ())
    lines = [f"Delivery -- {profile['display_name']}: "
             + " ".join(profile.get("delivery") or ())]
    if angles:
        lines.append(f"Angles: vary your angle every break -- {angles}. "
                     + _PALETTE)
    else:
        lines.append(_PALETTE)
    lines.append(MACHINERY_RULE)
    lines.append(_CHATTINESS_GUIDANCE.get(
        chattiness, _CHATTINESS_GUIDANCE[DEFAULT_CHATTINESS]))
    return "\n".join(lines) + "\n"


def ack_line(profile: dict[str, Any]) -> str:
    """The one delivery line a Start Radio liner carries."""
    return f"Delivery -- {profile['display_name']}: {profile.get('ack', '')}"


def catalog() -> list[dict[str, Any]]:
    """What a settings UI may show: ids, friendly names and descriptions.
    No guidance text -- prompt wording is not a user-facing surface -- and
    no chattiness default, because a personality no longer has one."""
    return [{"id": p["persona_id"], "name": p["display_name"],
             "description": p["description"],
             "voice_default": p["voice_default"]}
            for p in PROFILES.values()]
