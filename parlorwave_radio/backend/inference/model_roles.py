"""The canonical model-role vocabulary.

A role is a JOB a model does for ParlorWave, not a model. There are exactly
three, they are finite, and adding one is a deliberate documented change --
never something a package manifest or a config file can invent.

This module is the ONLY place the role strings are spelled. `llm_adapter`
re-exports them for its call sites; the package manifest and the role registry
validate against them. Nothing else may duplicate the literals.
"""

from __future__ import annotations

ROLE_DJ_WRITER = "dj_writer"
ROLE_STATION_PROGRAMMER = "station_programmer"
ROLE_SONG_CONTEXT_EXTRACTOR = "song_context_extractor"

# Ordered for stable serialization and for the adapter's public tuple.
ROLES: tuple[str, ...] = (
    ROLE_DJ_WRITER,
    ROLE_STATION_PROGRAMMER,
    ROLE_SONG_CONTEXT_EXTRACTOR,
)
ROLE_SET: frozenset[str] = frozenset(ROLES)


class UnknownRoleError(ValueError):
    """A role name outside the known set. Never coerced, never guessed."""

    def __init__(self, role: object) -> None:
        self.role = role
        super().__init__(f"unknown model role: {role!r} "
                         f"(known: {', '.join(ROLES)})")


def is_role(name: object) -> bool:
    return isinstance(name, str) and name in ROLE_SET


def validate_role(name: object) -> str:
    """Return `name` if it is a known role, else raise UnknownRoleError."""
    if not is_role(name):
        raise UnknownRoleError(name)
    return name  # type: ignore[return-value]
