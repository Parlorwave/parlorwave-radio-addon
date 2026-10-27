"""Central filesystem roots for Music Brain.

Every path the service touches derives from the two roots below, read
through the one configuration boundary (config.py). Consumers keep their
own module-level constants (musicdb.DB, playback.BASE, library_repo.INDEX,
...) derived from these roots at import time -- tests patch those
per-module constants directly and must continue to work.
"""

import config as _config

_cfg = _config.current()

# All service state: databases, music_index.json, caches, local config files.
BASE_DIR = _cfg.base_dir

# Root of a local music tree for index_library.py. None = no local library
# is configured (the ordinary product state; the catalog is not a filesystem).
MUSIC_ROOT = _cfg.music_root
