"""Which writer writes a break, and how that decision is recorded.

The packet (break_packet) is the only input any writer gets. Selection is
the smallest backward-compatible rule:

    an LLM writer is configured (or a test injects one)
        -> the LLM writes; deterministic copy is the fallback when it
           errors, returns nothing, or fails the authority guard
    no LLM writer configured
        -> deterministic copy is primary

One attempt each, in that order, never a retry loop: the LLM path keeps
its single repair rerender, the deterministic path gets no repair at all
(its copy is built from the packet, so a guard rejection there is a bug
worth failing loudly on, not retrying).
"""

from __future__ import annotations

from typing import Any

PROVIDER_OLLAMA = "ollama"
PROVIDER_DETERMINISTIC = "deterministic"
# Copy that was neither written by a model nor composed from the packet:
# a fixed station line, supplied whole by the caller. The news intro is
# the only one today. It has its own provider because the alternative is
# a label that is not true: the text reaches render_break through the
# llm_fn seam, and `llm_in_use` counts an occupied seam as an LLM.
PROVIDER_TEMPLATE = "template"

MODE_OLLAMA_PRIMARY = "ollama_primary"
MODE_DETERMINISTIC_PRIMARY = "deterministic_primary"
MODE_DETERMINISTIC_FALLBACK = "deterministic_fallback"
MODE_TEMPLATE_LINE = "template_line"
# The Smart DJ planner wrote the copy: one structured model call decided
# the KIND of remark (or silence) and the words together (smart_dj).
MODE_SMART_PLANNER = "smart_planner"
#: Music-aware stories: a story PREPARED and checked ahead of air from the
#: listener's Music Assistant biographies, assembled in code with the owed
#: time/weather and the handoff (split_hosting). No model call at air time.
MODE_STORY_PREPARED = "story_prepared"

REASON_PROVIDER_UNCONFIGURED = "provider_unconfigured"
REASON_PROVIDER_ERROR = "provider_error"
REASON_EMPTY_OUTPUT = "empty_output"
REASON_GUARD_REJECTED = "guard_rejected"
#: The draft was true and allowed, but did not pay this slot's REQUIRED
#: programming (required_hosting.unmet) -- so it does not get to satisfy
#: the hosted slot, and the deterministic floor hosts instead. Not a
#: grounding failure and never a reason to retry a model.
REASON_HOSTING_INCOMPLETE = "required_hosting_unmet"
#: The planner declined at enrichment level and the station still owes a
#: hosted transition, so the floor writes it. The planner's own reason
#: travels beside this one in telemetry and forensics.
REASON_PLANNER_DECLINED = "planner_declined"


def llm_in_use(llm_fn=None) -> bool:
    """True when an LLM writer should be tried first: an injected writer
    (tests, deps) always counts; otherwise only a writing model the
    runtime can ACTUALLY use.

    This is the single gate on both the ordinary writer and Smart DJ
    eligibility (dj_engine._planner_for), and it used to ask
    `config.current().writer`, which is read from the environment and
    nothing else. An owner who configured a model on the Settings screen
    -- address saved, model chosen, Test Connection green -- got
    `configured: false`, deterministic breaks and a planner that never
    once ran, because no OLLAMA_URL was set.

    `optional_services.effective_writer` answers for settings AND the
    environment, and is the same resolver the adapter uses to place the
    call and health uses to describe it, so the three cannot disagree.
    """
    if llm_fn is not None:
        return True
    try:
        import optional_services
        return bool(optional_services.effective_writer()["usable"])
    except Exception:  # noqa: BLE001 - an unreadable writer is no writer
        return False


def telemetry(provider: str, mode: str, *, fallback_reason: str | None = None,
              template_id: str | None = None,
              detail: str | None = None,
              kind: str | None = None,
              required_facts_used: list[str] | None = None) -> dict[str, Any]:
    """The break's 'writer' note: enough to tell WHY copy came from where.
    `detail` is a short, secret-free reason string (an error class, a
    guard category), never prompt or model text. `kind` is the Smart DJ
    planner's chosen kind (transition / quip / ...), recorded only for
    MODE_SMART_PLANNER."""
    out: dict[str, Any] = {"provider": provider, "mode": mode}
    if kind:
        out["kind"] = kind
    if mode == MODE_DETERMINISTIC_FALLBACK:
        out["fallback_from"] = PROVIDER_OLLAMA
        out["fallback_reason"] = fallback_reason
    if template_id:
        out["template_id"] = template_id
    if required_facts_used:
        # Which REQUIRED obligations the deterministic floor actually paid
        # (deterministic_writer facts_used). Read beside `fallback_reason`
        # it answers "the model failed this slot, and what did the station
        # manage instead" without re-reading the copy.
        out["required_facts_used"] = list(required_facts_used)
    if detail:
        out["detail"] = detail[:120]
    return out
